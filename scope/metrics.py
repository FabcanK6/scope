"""Evaluation metrics (pure Python, no seqeval / sklearn dependency)."""

from __future__ import annotations

from collections import Counter, defaultdict

from scope.schema import ACTION_TYPES, ISSUE_CODES, METADATA_TYPES, RISK_LEVELS
from scope.text import bio_to_spans


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def span_prf(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    """Exact-match span precision/recall/F1, micro-averaged and per span type."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold_tags, pred_tags):
        dummy = [""] * len(g)
        gs = {s.as_tuple() for s in bio_to_spans(dummy, g)}
        ps = {s.as_tuple() for s in bio_to_spans(dummy, p[:len(g)])}
        for s in gs & ps:
            tp[s[0]] += 1
        for s in ps - gs:
            fp[s[0]] += 1
        for s in gs - ps:
            fn[s[0]] += 1
    labels = sorted(set(tp) | set(fp) | set(fn))
    out = {"micro": prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))}
    out["per_type"] = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in labels}
    for name, group in (("metadata", METADATA_TYPES), ("actions", ACTION_TYPES)):
        out[name] = prf(sum(tp[x] for x in group), sum(fp[x] for x in group), sum(fn[x] for x in group))
    return out


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def action_prf(gold: list[list[dict]], pred: list[list[dict]]) -> dict:
    """An action item counts as correct only if action text, owner and due all match."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs = Counter((_norm(a["action"]), _norm(a["owner"]), _norm(a["due"])) for a in g)
        ps = Counter((_norm(a["action"]), _norm(a["owner"]), _norm(a["due"])) for a in p)
        hit = sum((gs & ps).values())
        tp += hit
        fp += sum(ps.values()) - hit
        fn += sum(gs.values()) - hit
    return prf(tp, fp, fn)


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total, ece = len(confidences), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            ece += len(idx) / total * abs(acc - conf)
    return ece


def evaluate_predictions(rows: list[dict], preds: list[dict], by_style: bool = True) -> dict:
    """``rows`` are dataset rows; ``preds`` hold risk, risk_probs, issues, tags (word level) and actions."""
    n = len(rows)
    # --- risk
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r, p in zip(rows, preds):
        confusion[r["risk"]][p["risk"]] += 1
    per_class = {}
    for lev in RISK_LEVELS:
        tp = confusion[lev][lev]
        fp = sum(confusion[o][lev] for o in RISK_LEVELS if o != lev)
        fn = sum(confusion[lev][o] for o in RISK_LEVELS if o != lev)
        per_class[lev] = prf(tp, fp, fn)
    risk_correct = [r["risk"] == p["risk"] for r, p in zip(rows, preds)]
    risk = {
        "accuracy": sum(risk_correct) / n if n else 0.0,
        "macro_f1": sum(m["f1"] for m in per_class.values()) / len(RISK_LEVELS),
        "per_class": per_class,
        "confusion": {g: dict(c) for g, c in confusion.items()},
        "high_risk_recall": per_class["high"]["recall"],
    }
    if all(p.get("risk_probs") for p in preds):
        conf = [float(p["risk_probs"][p["risk"]]) for p in preds]
        risk["ece"] = expected_calibration_error(conf, risk_correct)
        risk["mean_confidence"] = sum(conf) / n if n else 0.0

    # --- issues
    tp, fp, fn = Counter(), Counter(), Counter()
    exact_sets = 0
    for r, p in zip(rows, preds):
        g, q = set(r["issues"]), set(p["issues"])
        exact_sets += g == q
        for c in g & q:
            tp[c] += 1
        for c in q - g:
            fp[c] += 1
        for c in g - q:
            fn[c] += 1
    per_issue = {c: prf(tp[c], fp[c], fn[c]) for c in ISSUE_CODES}
    issues = {
        "micro": prf(sum(tp.values()), sum(fp.values()), sum(fn.values())),
        "macro_f1": sum(m["f1"] for m in per_issue.values()) / len(ISSUE_CODES),
        "exact_set_match": exact_sets / n if n else 0.0,
        "per_issue": per_issue,
    }
    traps = [(c in p["issues"]) for r, p in zip(rows, preds) for c in (r.get("inactive") or {})]
    if traps:
        issues["negated_mentions"] = len(traps)
        issues["false_alarm_on_negated"] = sum(traps) / len(traps)

    # --- spans and actions
    spans = span_prf([r["tags"] for r in rows], [p["tags"] for p in preds])
    actions = action_prf([r["actions"] for r in rows], [p["actions"] for p in preds])
    exact = sum(r["risk"] == p["risk"] and set(r["issues"]) == set(p["issues"]) and r["tags"] == p["tags"]
                for r, p in zip(rows, preds))

    result = {"n": n, "risk": risk, "issues": issues, "spans": spans, "actions": actions,
              "note_exact_match": exact / n if n else 0.0}
    if any("review_reasons" in p for p in preds):
        flagged = [bool(p.get("review_reasons")) for p in preds]
        ok = [c for c, f in zip(risk_correct, flagged) if not f]
        bad = [c for c, f in zip(risk_correct, flagged) if f]
        result["review"] = {"flag_rate": sum(flagged) / n if n else 0.0,
                            "risk_acc_unflagged": sum(ok) / len(ok) if ok else None,
                            "risk_acc_flagged": sum(bad) / len(bad) if bad else None,
                            "wrong_risk_caught": (sum(1 for c, f in zip(risk_correct, flagged) if f and not c)
                                                  / max(1, sum(1 for c in risk_correct if not c)))}
    if by_style:
        styles = sorted({r.get("style", "?") for r in rows})
        if len(styles) > 1:
            result["by_style"] = {}
            for st in styles:
                idx = [i for i, r in enumerate(rows) if r.get("style") == st]
                sub = evaluate_predictions([rows[i] for i in idx], [preds[i] for i in idx], by_style=False)
                result["by_style"][st] = {"n": len(idx), "risk_acc": sub["risk"]["accuracy"],
                                          "issue_f1": sub["issues"]["micro"]["f1"],
                                          "span_f1": sub["spans"]["micro"]["f1"], "action_f1": sub["actions"]["f1"]}
    return result


def headline(r: dict) -> dict:
    """The handful of numbers worth putting in a README table."""
    out = {
        "risk_acc": r["risk"]["accuracy"], "risk_macro_f1": r["risk"]["macro_f1"],
        "high_risk_recall": r["risk"]["high_risk_recall"],
        "issue_micro_f1": r["issues"]["micro"]["f1"], "span_f1": r["spans"]["micro"]["f1"],
        "metadata_f1": r["spans"]["metadata"]["f1"], "action_item_f1": r["actions"]["f1"],
        "note_exact": r["note_exact_match"],
    }
    if "false_alarm_on_negated" in r["issues"]:
        out["false_alarm_on_negated"] = r["issues"]["false_alarm_on_negated"]
    if "ece" in r["risk"]:
        out["risk_ece"] = r["risk"]["ece"]
    if "review" in r:
        out["review_flag_rate"] = r["review"]["flag_rate"]
        out["wrong_risk_flagged"] = r["review"]["wrong_risk_caught"]
    return out


def format_report(r: dict) -> str:
    lines = [
        f"notes                     : {r['n']}",
        f"risk accuracy             : {r['risk']['accuracy']:.3f}  (macro F1 {r['risk']['macro_f1']:.3f}, "
        f"high-risk recall {r['risk']['high_risk_recall']:.3f})",
    ]
    if "ece" in r["risk"]:
        lines.append(f"risk calibration (ECE)    : {r['risk']['ece']:.3f}  (mean confidence "
                     f"{r['risk']['mean_confidence']:.3f})")
    lines += [
        f"issue flags micro F1      : {r['issues']['micro']['f1']:.3f}  (macro {r['issues']['macro_f1']:.3f}, "
        f"exact set {r['issues']['exact_set_match']:.3f})",
    ]
    if "false_alarm_on_negated" in r["issues"]:
        lines.append(f"false alarms on negations : {r['issues']['false_alarm_on_negated']:.3f}  "
                     f"(share of {r['issues']['negated_mentions']} 'no issue' / 'resolved' mentions flagged anyway)")
    lines += [
        f"span F1 (all)             : {r['spans']['micro']['f1']:.3f}  (metadata {r['spans']['metadata']['f1']:.3f}, "
        f"action spans {r['spans']['actions']['f1']:.3f})",
        f"action items F1           : {r['actions']['f1']:.3f}  (action + owner + due all correct)",
        f"note exact match          : {r['note_exact_match']:.3f}",
        "",
        "per span type:",
    ]
    for lab, m in r["spans"]["per_type"].items():
        lines.append(f"  {lab:<11} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    lines.append("\nper issue:")
    for c, m in r["issues"]["per_issue"].items():
        lines.append(f"  {c:<19} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    lines.append("\nrisk confusion (rows = gold):")
    for g in RISK_LEVELS:
        row = r["risk"]["confusion"].get(g, {})
        lines.append(f"  {g:<7} " + "  ".join(f"{p}={row.get(p, 0):<4}" for p in RISK_LEVELS))
    if "review" in r:
        rv = r["review"]
        acc = ", ".join(f"{k}={v:.3f}" for k, v in rv.items() if v is not None)
        lines.append(f"\nneeds-review flag: {acc}")
    if "by_style" in r:
        lines.append("\nby note style:")
        for st, m in r["by_style"].items():
            lines.append(f"  {st:<10} n={m['n']:<5} risk={m['risk_acc']:.3f} issues={m['issue_f1']:.3f} "
                         f"spans={m['span_f1']:.3f} actions={m['action_f1']:.3f}")
    return "\n".join(lines)
