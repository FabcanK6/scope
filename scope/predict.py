"""Inference: site-visit note -> structured visit record.

    from scope.predict import load_parser
    parser = load_parser("models/scope-bert")      # hybrid; backend="bert" or "rules" for the others
    record = parser.analyze(note_text)
"""

from __future__ import annotations

from pathlib import Path

from scope.record import assemble_actions, build_record
from scope.rules import RuleParser
from scope.text import bio_to_spans, tokenize


def _with_actions(text: str, pred: dict) -> dict:
    pred["actions"] = [{k: a[k] for k in ("action", "owner", "due")} for a in assemble_actions(text, pred["spans"])]
    return pred


class RuleBasedParser(RuleParser):
    def predict(self, text: str) -> dict:
        return _with_actions(text, super().predict(text))

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def analyze(self, text: str) -> dict:
        return build_record(text, self.predict(text))


class BertParser:
    name = "bert"

    def __init__(self, model, tokenizer, cfg: dict, device):
        self.model, self.tokenizer, self.cfg, self.device = model, tokenizer, cfg, device
        self.max_length = cfg.get("max_length", 512)
        self.temperature = float(cfg.get("risk_temperature", 1.0))
        self.threshold = float(cfg.get("issue_threshold", 0.5))

    @classmethod
    def from_dir(cls, model_dir: str | Path, device: str = "auto") -> BertParser:
        from scope.model import ScopeModel, pick_device

        dev = pick_device(device)
        model, tokenizer, cfg = ScopeModel.load(model_dir, dev)
        return cls(model, tokenizer, cfg, dev)

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        import torch

        from scope.model import encode_words
        from scope.schema import ID2LABEL, ISSUE_CODES, RISK_LEVELS

        self.model.eval()
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            token_lists = [tokenize(t) for t in chunk]
            word_lists = [[t.text for t in toks] or ["."] for toks in token_lists]
            enc, firsts = encode_words(self.tokenizer, word_lists, self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            risk_p = (out["risk_logits"].float() / self.temperature).softmax(-1).cpu().tolist()
            issue_p = out["issue_logits"].float().sigmoid().cpu().tolist()
            tag_ids = out["tag_logits"].argmax(-1).cpu().tolist()
            for b, (text, toks) in enumerate(zip(chunk, token_lists)):
                tags = [ID2LABEL[tag_ids[b][t]] if t is not None else "O" for t in firsts[b]][:len(toks)]
                probs = dict(zip(RISK_LEVELS, risk_p[b]))
                issue_probs = dict(zip(ISSUE_CODES, issue_p[b]))
                pred = {
                    "tokens": toks, "tags": tags, "spans": bio_to_spans(toks, tags, text),
                    "risk": max(probs, key=probs.get), "risk_probs": probs,
                    "issues": [c for c in ISSUE_CODES if issue_probs[c] >= self.threshold],
                    "issue_probs": issue_probs, "truncated": any(t is None for t in firsts[b]),
                    "backend": self.name,
                }
                results.append(_with_actions(text, pred))
        return results

    def analyze(self, text: str) -> dict:
        return build_record(text, self.predict(text))

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [build_record(t, p) for t, p in zip(texts, self.predict_batch(texts))]


class HybridParser:
    """BERT for what needs understanding, rules for what follows a format.

    Evaluation showed a clear split: the fine-tuned model is far better at the
    judgement calls (risk level, which issues are active, negations), while the
    regular expressions are better at visit metadata and action-item lines in
    formats the model never saw (new date formats, new note layouts). So:

    * risk and issue flags come from BERT
    * metadata and action spans come from the rules
    * in any sentence where the rules found no action item, BERT's action spans are used
    """

    name = "hybrid"

    def __init__(self, bert: BertParser, rules: RuleBasedParser | None = None):
        self.bert = bert
        self.rules = rules or RuleBasedParser()

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        from scope.text import sentences

        out = []
        for text, b, r in zip(texts, self.bert.predict_batch(texts, batch_size), self.rules.predict_batch(texts)):
            tags = list(r["tags"])
            rule_action_sents = {i for i, (s, e) in enumerate(sentences(text))
                                 if any(sp.label == "ACTION" and s <= sp.char_start < e for sp in r["spans"])}
            for sp in b["spans"]:
                if sp.label not in ("ACTION", "OWNER", "DUE"):
                    continue
                sent = next((i for i, (s, e) in enumerate(sentences(text)) if s <= sp.char_start < e), None)
                if sent in rule_action_sents or any(t != "O" for t in tags[sp.start:sp.end]):
                    continue
                tags[sp.start:sp.end] = b["tags"][sp.start:sp.end]
            toks = r["tokens"]
            pred = {**b, "tags": tags, "spans": bio_to_spans(toks, tags, text), "backend": self.name}
            out.append(_with_actions(text, pred))
        return out

    def analyze(self, text: str) -> dict:
        return build_record(text, self.predict(text))


def load_parser(model_dir: str | Path | None = "models/scope-bert", device: str = "auto", backend: str = "hybrid"):
    """Return the requested parser. Falls back to the rule-based parser when no checkpoint exists.

    ``backend``: "hybrid" (default; BERT for risk and issues, rules for metadata), "bert" or "rules".
    """
    if backend != "rules" and model_dir and (Path(model_dir) / "scope_model.pt").exists():
        bert = BertParser.from_dir(model_dir, device)
        return HybridParser(bert) if backend == "hybrid" else bert
    return RuleBasedParser()
