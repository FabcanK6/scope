"""Compare SCOPE with Amazon Comprehend Medical on the same synthetic notes.

Comprehend Medical is a general-purpose medical NLP service. It finds medical
conditions, medications, procedures and protected health information (names,
dates, IDs) but has no notion of a site-visit record: no visit type, no issue
flags, no risk level, no action items. This script measures the overlap that
*can* be compared (dates and person names) and shows what each system returns.

Cost: Comprehend Medical bills per 100 characters (1 unit minimum per request).
The script prints an estimate and stops unless you pass --yes. Requires an AWS
account, `pip install boto3`, and credentials from `aws configure`.

    python scripts/compare_comprehend_medical.py --data data/test_unseen.jsonl --max-notes 50          # estimate only
    python scripts/compare_comprehend_medical.py --data data/test_unseen.jsonl --max-notes 50 --yes    # run it
    python scripts/compare_comprehend_medical.py --data handwritten --yes

Only ever send synthetic or properly de-identified text to a cloud API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.data.generate import read_jsonl  # noqa: E402
from scope.data.handwritten import load_handwritten  # noqa: E402
from scope.predict import load_parser  # noqa: E402
from scope.rules import DATE_RE  # noqa: E402

PRICE_PER_UNIT = 0.01  # USD, first tier for DetectEntitiesV2 (check current AWS pricing)
NAME_LABELS = {"MONITOR", "PI"}


def units(text: str) -> int:
    return max(1, math.ceil(len(text) / 100))


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score(gold: list[tuple[int, int]], pred: list[tuple[int, int]]) -> tuple[int, int, int]:
    tp = sum(any(overlaps(g, p) for p in pred) for g in gold)
    fp = sum(not any(overlaps(p, g) for g in gold) for p in pred)
    return tp, fp, len(gold) - tp


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/test_unseen.jsonl", help="JSONL file or 'handwritten'")
    ap.add_argument("--max-notes", type=int, default=50)
    ap.add_argument("--model", default="models/scope-bert")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--yes", action="store_true", help="actually call the AWS API (costs money)")
    ap.add_argument("--report", default="reports/comprehend_medical.json")
    args = ap.parse_args()

    rows = (load_handwritten() if args.data == "handwritten" else read_jsonl(args.data))[: args.max_notes]
    total_units = sum(units(r["text"]) for r in rows)
    print(f"{len(rows)} notes, {total_units} billing units, estimated cost ${total_units * PRICE_PER_UNIT:.2f} "
          "(before any free-tier credit)")
    if not args.yes:
        print("Dry run only. Re-run with --yes to call Comprehend Medical.")
        return

    import boto3

    client = boto3.client("comprehendmedical", region_name=args.region)
    parser = load_parser(args.model)
    print(f"SCOPE parser: {parser.name}")

    counts = {"cm_dates": [0, 0, 0], "scope_dates": [0, 0, 0], "cm_names": [0, 0, 0], "scope_names": [0, 0, 0]}
    categories, examples = Counter(), []
    for i, r in enumerate(rows):
        text = r["text"]
        resp = client.detect_entities_v2(Text=text)
        ents = resp.get("Entities", [])
        for e in ents:
            categories[f"{e['Category']}/{e['Type']}"] += 1
        cm_dates = [(e["BeginOffset"], e["EndOffset"]) for e in ents if e["Type"] == "DATE"]
        cm_names = [(e["BeginOffset"], e["EndOffset"]) for e in ents if e["Type"] == "NAME"]

        # gold: every date-like string in the note (visit date, due dates and others), and the named people
        gold_dates = [(m.start(), m.end()) for m in DATE_RE.finditer(text)]
        gold_names = [(s["char_start"], s["char_end"]) for s in r["spans"] if s["label"] in NAME_LABELS]

        pred = parser.predict(text)
        sc_dates = [(s.char_start, s.char_end) for s in pred["spans"] if s.label in ("VISIT_DATE", "DUE")
                    and DATE_RE.search(s.text)]
        sc_names = [(s.char_start, s.char_end) for s in pred["spans"] if s.label in NAME_LABELS]

        for key, gold, got in (("cm_dates", gold_dates, cm_dates), ("scope_dates", gold_dates, sc_dates),
                               ("cm_names", gold_names, cm_names), ("scope_names", gold_names, sc_names)):
            for j, v in enumerate(score(gold, got)):
                counts[key][j] += v
        if i < 3:
            examples.append({"id": r.get("id"), "comprehend_medical": [
                {k: e[k] for k in ("Text", "Category", "Type", "Score")} for e in ents],
                "scope": {"risk": pred["risk"], "issues": pred["issues"], "actions": pred["actions"]}})
        time.sleep(0.05)

    report = {k: prf(*v) for k, v in counts.items()}
    report["comprehend_medical_entity_types"] = dict(categories.most_common())
    report["examples"] = examples
    print("\n                      precision  recall    f1")
    for k in ("cm_dates", "scope_dates", "cm_names", "scope_names"):
        m = report[k]
        print(f"  {k:<18} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>6.3f}")
    print("\nComprehend Medical entity types found:")
    for t, n in categories.most_common(12):
        print(f"  {t:<45} {n}")
    print("\nIssue flags, risk level and action items: SCOPE only (no Comprehend Medical equivalent).")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
