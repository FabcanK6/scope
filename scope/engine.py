"""SCOPE's note-reading engine: a large language model reads, plain code verifies and scores.

    note ─► Gemini (instructions + severity rubric + 2 worked examples) ─► JSON:
              visit details, every finding (topic, status, severity, quoted evidence), action items
         ─► verification: every quote, name, date and action must be found word for word in the note
         ─► rubric: risk = high / medium / low from the verified active findings (deterministic code)
         ─► visit record: same structure as before, plus findings with evidence and a short summary

The LLM never decides the risk level directly and nothing it says is shown as a finding unless the
supporting text is actually in the note.
"""

from __future__ import annotations

import json
import re
import time

from scope.llm import RUBRIC_TEXT, SEVERITIES, STATUSES, GeminiClient, LLMError, score_findings, verify_quote
from scope.record import build_record, normalize_date
from scope.schema import ISSUE_BY_CODE, ISSUE_CODES
from scope.text import bio_to_spans, char_spans_to_bio, sentences, tokenize

_STR = {"type": "STRING", "nullable": True}
EXTRACT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "visit": {"type": "OBJECT", "properties": {
            "visit_type": _STR, "visit_date": _STR, "site": _STR, "monitor": _STR, "pi": _STR,
            "screened": _STR, "enrolled": _STR}},
        "findings": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "issue": {"type": "STRING", "enum": ISSUE_CODES},
            "status": {"type": "STRING", "enum": STATUSES},
            "severity": {"type": "STRING", "enum": SEVERITIES},
            "evidence": {"type": "STRING"},
            "explanation": {"type": "STRING"}},
            "required": ["issue", "status", "severity", "evidence", "explanation"]}},
        "actions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "owner": _STR, "action": {"type": "STRING"}, "due": _STR}, "required": ["action"]}},
        "summary": {"type": "STRING"},
    },
    "required": ["visit", "findings", "actions", "summary"],
}

ISSUE_LIST = "\n".join(f"- {c}: {ISSUE_BY_CODE[c].display} ({ISSUE_BY_CODE[c].group})" for c in ISSUE_CODES)

EXAMPLE_1 = """Monitor: Hannah Price, RN, CCRA
Visit Type: Interim Monitoring Visit (IMV)
Date: October 14, 2025
Notes:
Conducted a 100% source data verification (SDV) for Subject 004-12 and 004-15. Subject 004-12's informed consent form (ICF) was properly signed and dated prior to any protocol-specified procedures. However, a review of the concomitant medications log for Subject 004-15 revealed that the patient was prescribed Metoprolol by their primary care physician on September 3, 2025. This medication was not updated in the Electronic Case Report Form (eCRF). The Study Coordinator (SC) was retrained on the importance of real-time concomitant medication updates. The SC corrected the log during the visit, and I verified the entry against the source medical records. Investigational Product (IP) accountability was performed; the current inventory matches the interactive response technology (IRT) system logs exactly. No temperature excursions were noted on the digital data logger for the ambient storage closet."""  # noqa: E501

ANSWER_1 = {
    "visit": {"visit_type": "Interim Monitoring Visit", "visit_date": "October 14, 2025", "site": None,
              "monitor": "Hannah Price", "pi": None, "screened": None, "enrolled": None},
    "findings": [
        {"issue": "SDV_BACKLOG", "status": "no_issue", "severity": "minor",
         "evidence": "Conducted a 100% source data verification (SDV) for Subject 004-12 and 004-15.",
         "explanation": "SDV was completed for the subjects reviewed."},
        {"issue": "CONSENT", "status": "no_issue", "severity": "minor",
         "evidence": "Subject 004-12's informed consent form (ICF) was properly signed and dated prior to any "
                     "protocol-specified procedures.", "explanation": "Consent was done correctly."},
        {"issue": "DATA_ENTRY_BACKLOG", "status": "resolved_on_site", "severity": "minor",
         "evidence": "The SC corrected the log during the visit, and I verified the entry against the source "
                     "medical records.",
         "explanation": "A missing concomitant medication entry was fixed and verified during the visit."},
        {"issue": "IP_ACCOUNTABILITY", "status": "no_issue", "severity": "minor",
         "evidence": "the current inventory matches the interactive response technology (IRT) system logs exactly",
         "explanation": "Drug accountability reconciles."},
        {"issue": "TEMP_EXCURSION", "status": "no_issue", "severity": "minor",
         "evidence": "No temperature excursions were noted on the digital data logger for the ambient storage "
                     "closet.", "explanation": "Storage temperatures were fine."},
    ],
    "actions": [],
    "summary": "Routine interim visit with no open problems. A concomitant medication missing from the eCRF was "
               "corrected and verified on site, and the coordinator was retrained.",
}

EXAMPLE_2 = """Monitor: Rachel Moore, CCRC
Visit Type: Directed/For-Cause Monitoring Visit
Date: September 18, 2026
Notes:
This unscheduled visit was triggered due to a delay in the site reporting a Serious Adverse Event (SAE) for Subject 002-44.
The subject was hospitalized for acute cholecystitis on August 30, 2026, but the site did not notify the Sponsor until September 12, 2026, violating the mandatory 24-hour protocol reporting window. I met face-to-face with the PI to conduct a root-cause analysis.
I retrained both the PI and the primary SC on SAE definition and reporting timelines. The site has implemented a Corrective and Preventive Action (CAPA) plan. Site to confirm CAPA effectiveness by October 15, 2026."""  # noqa: E501

ANSWER_2 = {
    "visit": {"visit_type": "Directed/For-Cause Monitoring Visit", "visit_date": "September 18, 2026", "site": None,
              "monitor": "Rachel Moore", "pi": None, "screened": None, "enrolled": None},
    "findings": [
        {"issue": "SAE_REPORTING", "status": "active", "severity": "critical",
         "evidence": "the site did not notify the Sponsor until September 12, 2026, violating the mandatory "
                     "24-hour protocol reporting window",
         "explanation": "An SAE was reported 13 days late; late SAE reporting is critical even with a CAPA."},
    ],
    "actions": [{"owner": "Site", "action": "confirm CAPA effectiveness", "due": "October 15, 2026"}],
    "summary": "For-cause visit after a hospitalization was reported to the Sponsor 13 days late (critical). "
               "Staff were retrained and a CAPA is in place; the site must confirm it is effective.",
}

SYSTEM = f"""You read clinical trial site monitoring visit notes for a clinical research associate (CRA) and turn
each note into structured data.

Issue types:
{ISSUE_LIST}

{RUBRIC_TEXT}

Rules:
- List every topic from the issue types that the note mentions, including topics mentioned only to confirm they
  are fine (status "no_issue") and problems fixed and verified during the visit ("resolved_on_site").
- "evidence" must be copied word for word from the note: the shortest sentence or clause that shows it.
- Visit details (visit_type, visit_date, site, monitor, pi, screened, enrolled) must be copied word for word from
  the note, or null if the note does not state them. visit_type without an abbreviation in brackets. monitor and pi
  without credentials (RN, PhD, MD). screened and enrolled are the numbers only. Never use a date that is not the
  visit date (for example a hospitalization or prescription date) as visit_date.
- "actions" are open follow-ups still to be done. owner, action and due are copied word for word from the note
  (due may be null). Completed tasks, things already done during the visit, and general reminders are not actions.
- Do not invent anything. If the note does not say it, leave it out.
- "summary": two sentences a study manager can read in ten seconds.

Example note:
\"\"\"{EXAMPLE_1}\"\"\"
Answer: {json.dumps(ANSWER_1)}

Example note:
\"\"\"{EXAMPLE_2}\"\"\"
Answer: {json.dumps(ANSWER_2)}"""

# notes that appear in the prompt as worked examples; leave them out when evaluating
FEW_SHOT_IDS = {"real-01", "real-05"}


def _find(text: str, value: str | None, start: int = 0, end: int | None = None) -> tuple[int, int] | None:
    """Locate a value in the note: exact first, then ignoring case and runs of whitespace."""
    if not value or not str(value).strip():
        return None
    value = str(value).strip().strip(".,;")
    end = len(text) if end is None else end
    i = text.find(value, start, end)
    if i >= 0:
        return i, i + len(value)
    pattern = r"\s+".join(re.escape(w) for w in value.split())
    m = re.compile(pattern, re.I).search(text, start, end)
    return (m.start(), m.end()) if m else None


class LLMParser:
    name = "llm"

    def __init__(self, client: GeminiClient, sleep: float = 0.0):
        self.client = client
        self.sleep = sleep

    def extract(self, text: str) -> dict:
        return self.client.generate_json(SYSTEM, f"Note:\n\"\"\"\n{text}\n\"\"\"\nAnswer:", EXTRACT_SCHEMA)

    def predict(self, text: str) -> dict:
        data = self.extract(text)
        tokens = tokenize(text)
        char_spans: list[tuple[str, int, int]] = []
        problems: list[str] = []

        # visit details: keep only values that are really in the note
        v = data.get("visit") or {}
        date_at = re.search(r"(?:date of visit|visit date|date)\s*:", text, re.I)
        for key, label in (("visit_type", "VISIT_TYPE"), ("visit_date", "VISIT_DATE"), ("site", "SITE"),
                           ("monitor", "MONITOR"), ("pi", "PI"), ("screened", "SCREENED"), ("enrolled", "ENROLLED")):
            start = date_at.end() if (key == "visit_date" and date_at) else 0
            loc = _find(text, v.get(key), start) or _find(text, v.get(key))
            if loc:
                char_spans.append((label, *loc))
            elif v.get(key):
                problems.append(f"'{v.get(key)}' ({key.replace('_', ' ')}) is not in the note")

        # findings: verify the quoted evidence
        findings = []
        for f in data.get("findings") or []:
            if f.get("issue") not in ISSUE_CODES or f.get("status") not in STATUSES:
                continue
            if f.get("severity") not in SEVERITIES:
                f["severity"] = "minor"
            f["verified"] = verify_quote(f.get("evidence", ""), text)
            f["display"] = ISSUE_BY_CODE[f["issue"]].display
            f["group"] = ISSUE_BY_CODE[f["issue"]].group
            loc = _find(text, f.get("evidence"))
            f["char_start"], f["char_end"] = loc if loc else (-1, -1)
            findings.append(f)
        unverified = [f for f in findings if not f["verified"]]
        if unverified:
            problems.append(f"{len(unverified)} finding(s) quote text that is not in the note and were ignored")
        scored = score_findings(findings)

        # action items: the action text must be in the note; owner and due are looked for in the same sentence
        bounds = sentences(text)
        action_items = []
        for a in data.get("actions") or []:
            loc = _find(text, a.get("action"))
            if not loc:
                problems.append(f"action '{a.get('action')}' is not in the note")
                continue
            sent = next(((s, e) for s, e in bounds if s <= loc[0] < e), (0, len(text)))
            owner = _find(text, a.get("owner"), sent[0], sent[1])
            due = _find(text, a.get("due"), sent[0], sent[1])
            char_spans.append(("ACTION", *loc))
            for lab, span in (("OWNER", owner), ("DUE", due)):
                if span:
                    char_spans.append((lab, *span))
            due_text = text[due[0]:due[1]] if due else None
            action_items.append({"action": text[loc[0]:loc[1]], "owner": text[owner[0]:owner[1]] if owner else None,
                                 "due": due_text, "due_date": normalize_date(due_text)})

        # drop overlapping spans (keep the first) before tagging words
        kept: list[tuple[str, int, int]] = []
        for sp in sorted(char_spans, key=lambda s: s[1]):
            if all(sp[2] <= k[1] or sp[1] >= k[2] for k in kept):
                kept.append(sp)
        tags = char_spans_to_bio(tokens, kept)
        issues = [c for c in ISSUE_CODES if c in scored["active"]]
        return {
            "tokens": tokens, "tags": tags, "spans": bio_to_spans(tokens, tags, text),
            "risk": scored["risk"], "risk_probs": {}, "issues": issues, "issue_probs": {c: 1.0 for c in issues},
            "severities": scored["active"], "points": scored["points"], "findings": findings,
            "action_items": action_items, "actions": [{k: a[k] for k in ("action", "owner", "due")}
                                                      for a in action_items],
            "summary": data.get("summary", ""), "review_reasons": problems, "backend": self.name,
            "model": self.client.model,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        out = []
        for i, t in enumerate(texts):
            if i and self.sleep:
                time.sleep(self.sleep)
            out.append(self.predict(t))
        return out

    def analyze(self, text: str) -> dict:
        return build_record(text, self.predict(text))


__all__ = ["LLMParser", "LLMError", "SYSTEM", "FEW_SHOT_IDS"]
