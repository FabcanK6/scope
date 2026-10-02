"""Fine-tune a BERT encoder for joint risk + issue + span prediction.

Usage:
    python -m scope.train --model bert-base-uncased --out models/scope-bert --fp16
    python -m scope.train --model emilyalsentzer/Bio_ClinicalBERT --out models/scope-clinicalbert --fp16
    python -m scope.train --model prajjwal1/bert-tiny --epochs 1 --limit 300 --out models/scope-tiny  # smoke test

After training, the best checkpoint is calibrated with temperature scaling on
the validation set, so the risk confidence shown in the app is closer to the
real hit rate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from scope.data.generate import read_jsonl
from scope.metrics import evaluate_predictions, format_report
from scope.model import CONFIG_NAME, ScopeModel, encode_words, fit_temperature, pick_device
from scope.predict import BertParser
from scope.schema import ISSUE2ID, ISSUE_CODES, RISK2ID

PREFIX_SPACE_MODELS = {"roberta", "deberta", "deberta-v2", "longformer", "gpt2", "bart"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(name: str):
    model_type = AutoConfig.from_pretrained(name).model_type
    kwargs = {"add_prefix_space": True} if model_type in PREFIX_SPACE_MODELS else {}
    return AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)


def make_collate(tokenizer, max_length: int):
    def collate(rows: list[dict]):
        enc, _ = encode_words(tokenizer, [r["tokens"] for r in rows], max_length, [r["tags"] for r in rows])
        enc["risk_labels"] = torch.tensor([RISK2ID[r["risk"]] for r in rows], dtype=torch.long)
        issues = torch.zeros(len(rows), len(ISSUE_CODES))
        for i, r in enumerate(rows):
            for c in r["issues"]:
                issues[i, ISSUE2ID[c]] = 1.0
        enc["issue_labels"] = issues
        return dict(enc)
    return collate


@torch.no_grad()
def risk_logits(model, tokenizer, rows: list[dict], max_length: int, device, batch_size: int = 32):
    model.eval()
    logits, labels = [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc, _ = encode_words(tokenizer, [r["tokens"] for r in batch], max_length)
        out = model(**{k: v.to(device) for k, v in enc.items()})
        logits.append(out["risk_logits"].float().cpu())
        labels += [RISK2ID[r["risk"]] for r in batch]
    return torch.cat(logits), torch.tensor(labels)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--model", default="bert-base-uncased", help="any Hugging Face encoder with a fast tokenizer")
    ap.add_argument("--out", default="models/scope-bert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="learning rate for the three linear heads")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows (debugging)")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    train_rows = read_jsonl(args.train)[: args.limit]
    val_rows = read_jsonl(args.val)
    tokenizer = load_tokenizer(args.model)
    model = ScopeModel.from_encoder_name(args.model, dropout=args.dropout).to(device)

    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(tokenizer, args.max_length))

    heads = [model.risk_head, model.issue_head, model.tag_head]
    head_params = [p for h in heads for p in h.parameters()]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    enc_named = list(model.encoder.named_parameters())
    groups = [
        {"params": [p for n, p in enc_named if not any(nd in n for nd in no_decay)],
         "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [p for n, p in enc_named if any(nd in n for nd in no_decay)], "lr": args.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(groups)
    total_steps = math.ceil(len(loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, history = -1.0, []
    out_dir = Path(args.out)
    cfg = {"max_length": args.max_length}
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(**batch)
            optim.zero_grad(set_to_none=True)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += out["loss"].item()
            if step % 50 == 0 or step == len(loader):
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"(risk {out['risk_loss']:.3f} issues {out['issue_loss']:.3f} tags {out['tag_loss']:.3f})",
                      flush=True)

        parser = BertParser(model, tokenizer, cfg, device)
        preds = parser.predict_batch([r["text"] for r in val_rows], batch_size=32)
        report = evaluate_predictions(val_rows, preds)
        score = (report["risk"]["accuracy"] + report["issues"]["micro"]["f1"] + report["spans"]["micro"]["f1"]) / 3
        history.append({"epoch": epoch, "train_loss": running / len(loader), "val_risk_acc": report["risk"]["accuracy"],
                        "val_issue_f1": report["issues"]["micro"]["f1"], "val_span_f1": report["spans"]["micro"]["f1"],
                        "val_action_f1": report["actions"]["f1"], "seconds": round(time.time() - t0, 1)})
        print(f"\n== epoch {epoch} validation ==\n{format_report(report)}\n", flush=True)
        if score > best_score:
            best_score = score
            model.save(out_dir, tokenizer, base_model=args.model, max_length=args.max_length,
                       extra={"val": {k: v for k, v in history[-1].items() if k.startswith("val_")}, "epoch": epoch})
            print(f"saved best checkpoint to {out_dir}/ (score {score:.4f})")

    # --- calibrate the risk head on the validation set (temperature scaling)
    best, _tok, _cfg = ScopeModel.load(out_dir, device)
    logits, labels = risk_logits(best, tokenizer, val_rows, args.max_length, device)
    temperature = fit_temperature(logits, labels)
    cfg_path = out_dir / CONFIG_NAME
    saved = json.loads(cfg_path.read_text())
    saved["risk_temperature"] = round(temperature, 4)
    cfg_path.write_text(json.dumps(saved, indent=2))
    print(f"risk temperature fitted on validation: T={temperature:.3f} (T > 1 means the raw model was overconfident)")

    (out_dir / "training_history.json").write_text(json.dumps({"args": vars(args), "history": history,
                                                               "risk_temperature": temperature}, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
