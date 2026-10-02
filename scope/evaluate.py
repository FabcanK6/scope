"""Evaluate a parser on one or more test sets.

    python -m scope.evaluate --backend rules
    python -m scope.evaluate --model models/scope-bert --report reports/bert.json
    python -m scope.evaluate --model models/scope-bert --data data/test_unseen.jsonl --errors 5

``--data`` takes JSONL files and/or the word ``handwritten`` (the built-in set of
hand-written notes). The default is all three test sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scope.data.generate import read_jsonl
from scope.data.handwritten import load_handwritten
from scope.metrics import evaluate_predictions, format_report, headline

DEFAULT_SETS = ["data/test.jsonl", "data/test_unseen.jsonl", "handwritten"]


def load_set(name: str) -> list[dict]:
    return load_handwritten() if name == "handwritten" else read_jsonl(name)


def show_errors(rows: list[dict], preds: list[dict], k: int) -> None:
    shown = 0
    for r, p in zip(rows, preds):
        if shown >= k:
            break
        if r["risk"] == p["risk"] and set(r["issues"]) == set(p["issues"]):
            continue
        shown += 1
        print(f"\n--- {r.get('id')} ({r.get('style')}) ---")
        print(r["text"])
        print(f"  gold risk={r['risk']:<6} issues={sorted(r['issues'])}")
        conf = p["risk_probs"].get(p["risk"]) if p.get("risk_probs") else None
        extra = f"  (risk conf {conf:.2f})" if conf else ""
        print(f"  pred risk={p['risk']:<6} issues={sorted(p['issues'])}{extra}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--model", default="models/scope-bert")
    ap.add_argument("--backend", choices=["auto", "hybrid", "bert", "rules"], default="auto",
                    help="auto = plain BERT if a checkpoint exists, else rules")
    ap.add_argument("--report", default=None, help="write the full metrics to this JSON file")
    ap.add_argument("--errors", type=int, default=0, help="print this many misclassified notes per set")
    ap.add_argument("--quiet", action="store_true", help="only print the summary table")
    args = ap.parse_args(argv)

    from scope.predict import BertParser, HybridParser, RuleBasedParser, load_parser

    if args.backend == "rules":
        parser = RuleBasedParser()
    elif args.backend == "bert":
        parser = BertParser.from_dir(args.model)
    elif args.backend == "hybrid":
        parser = HybridParser(BertParser.from_dir(args.model))
    else:
        parser = load_parser(args.model, backend="bert")
    print(f"parser: {parser.name}")

    reports, table = {}, []
    for name in args.data:
        if name != "handwritten" and not Path(name).exists():
            print(f"skipping {name} (not found)")
            continue
        rows = load_set(name)
        preds = parser.predict_batch([r["text"] for r in rows])
        rep = evaluate_predictions(rows, preds)
        label = Path(name).stem if name != "handwritten" else "handwritten"
        reports[label] = rep
        table.append((label, headline(rep)))
        if not args.quiet:
            print(f"\n===== {label} =====\n{format_report(rep)}")
        if args.errors:
            show_errors(rows, preds, args.errors)

    if table:
        keys = list(table[0][1])
        print("\n" + "set".ljust(14) + "".join(k[:14].rjust(15) for k in keys))
        for label, h in table:
            print(label.ljust(14) + "".join(f"{h.get(k, float('nan')):15.3f}" for k in keys))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({"parser": parser.name, "sets": reports}, indent=2))
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
