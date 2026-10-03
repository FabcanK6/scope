"""One BERT encoder with three heads, trained together.

    note words ─► WordPiece ─► BERT encoder ──┬─► [CLS] ─► Linear ─► risk level (3 classes, softmax)
                                              ├─► [CLS] ─► Linear ─► issue flags (12 labels, sigmoid)
                                              └─► tokens ─► Linear ─► BIO tag per word (metadata + actions)

Loss = cross-entropy(risk) + binary cross-entropy(issues) + cross-entropy(tags).
Any Hugging Face encoder with a fast tokenizer works, e.g. ``bert-base-uncased``
or ``emilyalsentzer/Bio_ClinicalBERT``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scope.schema import BIO_LABELS, V1_ISSUE_CODES, LABEL2ID, RISK_LEVELS

WEIGHTS_NAME = "scope_model.pt"
CONFIG_NAME = "scope_config.json"


class ScopeModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1, issue_loss_weight: float = 1.0,
                 tag_loss_weight: float = 1.0):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(hidden, len(RISK_LEVELS))
        self.issue_head = nn.Linear(hidden, len(V1_ISSUE_CODES))
        self.tag_head = nn.Linear(hidden, len(BIO_LABELS))
        self.issue_loss_weight = issue_loss_weight
        self.tag_loss_weight = tag_loss_weight
        self._accepts_token_types = "token_type_ids" in inspect.signature(encoder.forward).parameters

    @classmethod
    def from_encoder_name(cls, name: str, **kwargs) -> ScopeModel:
        return cls(AutoModel.from_pretrained(name), **kwargs)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                risk_labels=None, issue_labels=None, tag_labels=None) -> dict:
        enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self._accepts_token_types:
            enc_kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**enc_kwargs).last_hidden_state           # (B, T, H)
        cls_vec = self.dropout(hidden[:, 0])
        out = {
            "risk_logits": self.risk_head(cls_vec),                       # (B, 3)
            "issue_logits": self.issue_head(cls_vec),                     # (B, 12)
            "tag_logits": self.tag_head(self.dropout(hidden)),            # (B, T, L)
        }
        if risk_labels is not None and issue_labels is not None and tag_labels is not None:
            risk_loss = nn.functional.cross_entropy(out["risk_logits"], risk_labels)
            issue_loss = nn.functional.binary_cross_entropy_with_logits(out["issue_logits"], issue_labels.float())
            tl = out["tag_logits"]
            tag_loss = nn.functional.cross_entropy(tl.reshape(-1, tl.size(-1)), tag_labels.reshape(-1),
                                                   ignore_index=-100)
            out.update(loss=risk_loss + self.issue_loss_weight * issue_loss + self.tag_loss_weight * tag_loss,
                       risk_loss=risk_loss.detach(), issue_loss=issue_loss.detach(), tag_loss=tag_loss.detach())
        return out

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path, tokenizer, base_model: str, max_length: int, extra: dict | None = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(out)
        tokenizer.save_pretrained(out)
        torch.save(self.state_dict(), out / WEIGHTS_NAME)
        cfg = {
            "base_model": base_model,
            "risk_levels": RISK_LEVELS,
            "issue_codes": V1_ISSUE_CODES,
            "bio_labels": BIO_LABELS,
            "max_length": max_length,
            "dropout": self.dropout.p,
            "risk_temperature": 1.0,
            "issue_threshold": 0.5,
            **(extra or {}),
        }
        (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path, device: str | torch.device = "cpu"):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / CONFIG_NAME).read_text())
        if (cfg["risk_levels"], cfg["issue_codes"], cfg["bio_labels"]) != (RISK_LEVELS, V1_ISSUE_CODES, BIO_LABELS):
            raise ValueError("Checkpoint label set differs from scope.schema; retrain or restore the old schema.")
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
        model = cls(encoder, dropout=cfg.get("dropout", 0.1))
        state = torch.load(model_dir / WEIGHTS_NAME, map_location=device, weights_only=True)
        model.load_state_dict(state)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model.to(device).eval(), tokenizer, cfg


def encode_words(tokenizer, batch_words: list[list[str]], max_length: int,
                 batch_tags: list[list[str]] | None = None):
    """Tokenize pre-split words and align word-level BIO tags to WordPiece tokens.

    Only the first sub-token of each word carries a label; the rest (and the
    special tokens) get -100 so the loss ignores them. Returns the encoding and,
    per example, the token index of each word's first sub-token (or None when
    the word was cut off by ``max_length``).
    """
    if not tokenizer.is_fast:
        raise ValueError("SCOPE needs a *fast* tokenizer (word_ids()).")
    enc = tokenizer(batch_words, is_split_into_words=True, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt")
    first_index: list[list[int | None]] = []
    labels = []
    for b, words in enumerate(batch_words):
        word_ids = enc.word_ids(batch_index=b)
        firsts: list[int | None] = [None] * len(words)
        row = [-100] * len(word_ids)
        prev = None
        for t, w in enumerate(word_ids):
            if w is not None and w != prev:
                firsts[w] = t
                if batch_tags is not None:
                    row[t] = LABEL2ID[batch_tags[b][w]]
            prev = w
        first_index.append(firsts)
        labels.append(row)
    if batch_tags is not None:
        enc["tag_labels"] = torch.tensor(labels, dtype=torch.long)
    return enc, first_index


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T that minimizes NLL on validation logits."""
    log_t = torch.zeros(1, requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.05, max_iter=steps)
    logits, labels = logits.detach().float().cpu(), labels.detach().cpu()

    def closure():
        optim.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_t.exp().item())
