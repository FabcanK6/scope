"""Teacher data: let an LLM write realistic notes for scenarios whose labels we already know.

    GEMINI_API_KEY=... python scripts/generate_teacher_notes.py --n 200 --out data/teacher.jsonl

For every note:
1. SCOPE's generator draws a scenario: visit type, date, site, people, which findings are active (and how
   severe), which topics are fine or were fixed on site, and the open action items.
2. Gemini writes the note in a varied, realistic style and marks the metadata and action items inline.
3. The result is kept only if it checks out: the LLM's own list of findings must match the scenario
   (same active issues, same severities), every quoted sentence must be in the note, and the markup must parse.

Labels therefore come from the scenario, not from the LLM's judgement; the LLM only supplies the wording.
The script appends to ``--out`` and can be stopped and restarted. It sleeps between requests to stay inside
free-tier limits. Train with: python -m scope.train --extra-train data/teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.data.generate import sample_note  # noqa: E402
from scope.data.handwritten import parse_marked  # noqa: E402
from scope.llm import RUBRIC_TEXT, GeminiClient, LLMError, get_api_key, verify_quote  # noqa: E402
from scope.record import assemble_actions  # noqa: E402
from scope.schema import ENTITY_TYPES, ISSUE_BY_CODE, ISSUE_CODES  # noqa: E402
from scope.text import bio_to_spans, char_spans_to_bio, tokenize  # noqa: E402

STYLES = [
    "a formal monitoring visit report with a header block (Monitor, Visit Type, Date) and full paragraphs",
    "a formal report with topic bullets such as 'IP Accountability:' and 'Regulatory Binder:'",
    "quick field notes typed on a phone right after the visit: abbreviations, fragments, lowercase",
    "an e-mail to the study team summarizing the visit",
    "a narrative paragraph written by an experienced CRA, with some long sentences",
]
AREAS = ["oncology", "type 2 diabetes", "heart failure", "rheumatoid arthritis", "asthma", "Alzheimer's disease",
         "chronic kidney disease", "a rare metabolic disorder", "psoriasis", "major depressive disorder"]

SYSTEM = f"""You write realistic clinical trial site monitoring visit notes as an experienced clinical research
associate would write them. You are given the facts of one visit and must express exactly those facts, no more.

{RUBRIC_TEXT}

Markup: wrap these items in the note with [[text|LABEL]] exactly as they appear: the visit type (VISIT_TYPE),
the visit date (VISIT_DATE), the site (SITE), the monitor's name (MONITOR), the PI's name (PI), the number of
subjects screened (SCREENED) and enrolled or randomized (ENROLLED), and for each open action item: who owns it
(OWNER), what must be done (ACTION) and when (DUE). Keep owner, action and due date of one action item in the same
sentence. Do not mark anything else. Completed tasks are not action items.

Return JSON with "note" (the marked-up note) and "findings": one entry per topic you wrote about, with the issue
code, status (active / resolved_on_site / no_issue), severity and the sentence from the note (without markup)
that states it."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "note": {"type": "STRING"},
        "findings": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "issue": {"type": "STRING", "enum": ISSUE_CODES},
            "status": {"type": "STRING", "enum": ["active", "resolved_on_site", "no_issue"]},
            "severity": {"type": "STRING", "enum": ["minor", "major", "critical"]},
            "evidence": {"type": "STRING"}},
            "required": ["issue", "status", "severity", "evidence"]}},
    },
    "required": ["note", "findings"],
}


def scenario_prompt(row: dict, rng: random.Random) -> str:
    m = row["meta"]
    lines = [
        f"Style: {rng.choice(STYLES)}. Therapeutic area: {rng.choice(AREAS)}.",
        f"Visit type: {m['visit_type']} | visit date: {m['visit_date']} | site number: {m['site_id']} | "
        f"monitor: {m['monitor']} | PI: {m['pi']}",
    ]
    if m.get("screened") is not None:
        lines.append(f"Screened: {m['screened']}, enrolled: {m['enrolled']} (mention both numbers).")
    lines.append("Active findings (problems that still exist after the visit):")
    lines += [f"- {c} ({ISSUE_BY_CODE[c].display}), severity {row['severities'][c]}" for c in row["issues"]]
    if not row["issues"]:
        lines.append("- none: this is a clean visit")
    lines.append("Topics that are fine or were corrected and verified during the visit:")
    lines += [f"- {c} ({ISSUE_BY_CODE[c].display}): {'confirmed fine' if k == 'negated' else 'corrected on site'}"
              for c, k in row["inactive"].items()]
    lines.append("Open action items (owner / action / due):")
    lines += [f"- {a['owner']} / {a['action']} / {a['due'] or 'no due date'}" for a in row["actions"]]
    if not row["actions"]:
        lines.append("- none")
    lines.append("You may add neutral details (arrival time, training, supplies) that are not findings.")
    return "\n".join(lines)


def check(row: dict, data: dict) -> dict | None:
    """Turn the LLM's answer into a training row, or None if it does not match the scenario."""
    try:
        text, char_spans = parse_marked(data["note"])
    except (KeyError, TypeError):
        return None
    if "[[" in text or "]]" in text or any(lab not in ENTITY_TYPES for lab, _, _ in char_spans):
        return None
    active = {f["issue"]: f["severity"] for f in data.get("findings", []) if f.get("status") == "active"}
    if active != row["severities"]:
        return None  # the LLM added, dropped or re-graded a finding
    if not all(verify_quote(f.get("evidence", ""), text) for f in data["findings"]):
        return None
    tokens = tokenize(text)
    for _lab, cs, ce in char_spans:  # spans must line up with whole tokens
        covered = [t for t in tokens if t.start >= cs and t.end <= ce]
        if not covered or covered[0].start != cs or covered[-1].end != ce:
            return None
    tags = char_spans_to_bio(tokens, char_spans)
    if not any(lab == "VISIT_DATE" for lab, _, _ in char_spans):
        return None
    actions = [{k: a[k] for k in ("action", "owner", "due")}
               for a in assemble_actions(text, bio_to_spans(tokens, tags, text))]
    inactive = {f["issue"]: "negated" if f["status"] == "no_issue" else "resolved"
                for f in data["findings"] if f.get("status") != "active" and f["issue"] not in active}
    return {**{k: row[k] for k in ("risk", "issues", "severities", "meta")}, "inactive": inactive,
            "text": text, "tokens": [t.text for t in tokens], "tags": tags, "actions": actions,
            "spans": [{"label": lab, "char_start": s, "char_end": e, "text": text[s:e]} for lab, s, e in char_spans],
            "style": "llm", "unseen": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of accepted notes to add")
    ap.add_argument("--out", default="data/teacher.jsonl")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=6.0, help="seconds between requests (free-tier friendly)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = GeminiClient(get_api_key() or "", model=args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in out.open()) if out.exists() else 0
    rng = random.Random(args.seed + have)
    accepted = rejected = 0
    while accepted < args.n:
        row = sample_note(rng)
        try:
            data = client.generate_json(SYSTEM, scenario_prompt(row, rng), SCHEMA)
        except LLMError as e:
            print(f"stopping: {e}")
            break
        good = check(row, data)
        if good:
            good["id"] = f"teacher-{have + accepted:05d}"
            with out.open("a") as f:
                f.write(json.dumps(good) + "\n")
            accepted += 1
        else:
            rejected += 1
        print(f"accepted {accepted}/{args.n}  rejected {rejected}  (model {client.model})", flush=True)
        time.sleep(args.sleep)
    print(f"done: {have + accepted} teacher notes in {out}")


if __name__ == "__main__":
    main()
