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

import copy
import json
import re
import time

from scope.llm import ESCALATION_SUBJECTS, RUBRIC_TEXT, BadAnswer, _norm, final_severity, SEVERITIES, STATUSES, GeminiClient, LLMError, score_findings, verify_quote
from scope.record import build_record, normalize_date
from scope.rules import RuleParser
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
            "explanation": {"type": "STRING"},
            "repeat": {"type": "BOOLEAN"},
            "subjects_affected": {"type": "INTEGER", "nullable": True},
            "site_wide": {"type": "BOOLEAN"},
            "escalation_evidence": {"type": "STRING"}},
            "required": ["issue", "status", "severity", "evidence", "explanation", "repeat", "subjects_affected",
                         "site_wide", "escalation_evidence"],
            "propertyOrdering": ["issue", "status", "severity", "evidence", "explanation", "repeat",
                                 "subjects_affected", "site_wide", "escalation_evidence"]}},
        "actions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "owner": _STR, "action": {"type": "STRING"}, "due": _STR}, "required": ["action"]}},
        "summary": {"type": "STRING"},
    },
    "required": ["visit", "findings", "actions", "summary"],
    "propertyOrdering": ["findings", "actions", "summary", "visit"],
}

ISSUE_LIST = "\n".join(f"- {c}: {ISSUE_BY_CODE[c].display} ({ISSUE_BY_CODE[c].group})" for c in ISSUE_CODES)

# Worked examples for the prompt. They are written for the prompt only and do not appear in any test set
# or in the app's example notes, so evaluation and the demo buttons are fair.
EXAMPLE_1 = """Site: 231 - Lakeview Clinical Research
CRA: Tomas Ferreira, CCRA
Visit Type: Interim Monitoring Visit (IMV)
Date of Visit: 12-Mar-2026
Screened: 5 / Randomized: 2 (target: 8 by end of March)
Observations:
Reviewed source documents for Subjects 231-003 through 231-006. All four subjects were re-consented on ICF version 4.0 within the timeframe required by the IRB. The delegation of authority log was missing the start date for the new Sub-Investigator; the coordinator added the date and the PI initialed the entry while I was on site. Two data queries from the previous visit were answered and closed during the visit. Dispensing records in the pharmacy binder reconcile with the returned kits. Enrollment remains well behind target. PI to submit an enrollment recovery plan to the Sponsor by April 3, 2026."""  # noqa: E501

ANSWER_1 = {
    "visit": {"visit_type": "Interim Monitoring Visit (IMV)", "visit_date": "12-Mar-2026",
              "site": "231 - Lakeview Clinical Research", "monitor": "Tomas Ferreira", "pi": None,
              "screened": "5", "enrolled": "2"},
    "findings": [
        {"issue": "CONSENT", "status": "no_issue", "severity": "minor",
         "evidence": "All four subjects were re-consented on ICF version 4.0 within the timeframe required by the IRB.",
         "explanation": "Re-consent was done on time."},
        {"issue": "STAFF_TURNOVER", "status": "resolved_on_site", "severity": "minor",
         "evidence": "the coordinator added the date and the PI initialed the entry while I was on site",
         "explanation": "A missing delegation log date was fixed during the visit."},
        {"issue": "QUERY_AGING", "status": "resolved_on_site", "severity": "minor",
         "evidence": "Two data queries from the previous visit were answered and closed during the visit.",
         "explanation": "Open queries were closed on site."},
        {"issue": "IP_ACCOUNTABILITY", "status": "no_issue", "severity": "minor",
         "evidence": "Dispensing records in the pharmacy binder reconcile with the returned kits.",
         "explanation": "Drug accountability reconciles."},
        {"issue": "ENROLLMENT_LAG", "status": "active", "severity": "major",
         "evidence": "Enrollment remains well behind target.",
         "explanation": "2 randomized against a target of 8; enrollment far behind target is major."},
    ],
    "actions": [{"owner": "PI", "action": "submit an enrollment recovery plan to the Sponsor",
                 "due": "April 3, 2026"}],
    "summary": "Interim visit with one open problem: enrollment is well behind target (2 of 8), and the PI owes a "
               "recovery plan by April 3. A delegation log gap and two old queries were fixed on site.",
}

EXAMPLE_2 = """IMV - Site 408 - 09/22/2026 - CRA: Priya Nair
- Site manager refused to give me read access to the hospital EMR for Subject 408-011 (says new hospital policy). Could not verify the Week 8 visit or any of the AE source for this subject.
- Pharmacy temp logs reviewed through 21-Sep, all within range.
- 3 queries open > 30 days on the Week 4 labs page, the same ones I flagged at the August visit.
- Week 4 vital signs not yet entered in EDC for Subjects 408-002, 408-005 and 408-009.
- Follow-up: PI to arrange EMR access for the monitor before the next visit on 10/20/2026. Escalated to the Sponsor study manager today."""  # noqa: E501

ANSWER_2 = {
    "visit": {"visit_type": "IMV", "visit_date": "09/22/2026", "site": "Site 408", "monitor": "Priya Nair",
              "pi": None, "screened": None, "enrolled": None},
    "findings": [
        {"issue": "SDV_BACKLOG", "status": "active", "severity": "critical",
         "evidence": "Site manager refused to give me read access to the hospital EMR for Subject 408-011",
         "explanation": "The site refused access to source documents, so this subject's data cannot be verified.",
         "subjects_affected": 1},
        {"issue": "TEMP_EXCURSION", "status": "no_issue", "severity": "minor",
         "evidence": "Pharmacy temp logs reviewed through 21-Sep, all within range.",
         "explanation": "Storage temperatures were fine."},
        {"issue": "QUERY_AGING", "status": "active", "severity": "minor",
         "evidence": "3 queries open > 30 days on the Week 4 labs page",
         "explanation": "A few aging queries remain open; they were already flagged at the previous visit.",
         "repeat": True, "escalation_evidence": "the same ones I flagged at the August visit"},
        {"issue": "DATA_ENTRY_BACKLOG", "status": "active", "severity": "minor",
         "evidence": "Week 4 vital signs not yet entered in EDC for Subjects 408-002, 408-005 and 408-009.",
         "explanation": "Vital signs pages are missing for three subjects.",
         "subjects_affected": 3, "escalation_evidence": "for Subjects 408-002, 408-005 and 408-009"},
    ],
    "actions": [{"owner": "PI", "action": "arrange EMR access for the monitor",
                 "due": "before the next visit on 10/20/2026"}],
    "summary": "The site refused the monitor access to source records for one subject (critical) and the issue has "
               "been escalated; the PI must restore EMR access before the next visit. Aging queries are a repeat "
               "finding and vital signs are missing for three subjects.",
}

EXAMPLE_3 = """Hi Jen, quick recap of today's remote visit (Oct 2, 2026) for site 552. eCRF pages are current and SDV is up to date through Visit 6. One subject's Week 12 visit fell two days outside the window because of a holiday closure; the site has logged it as a minor deviation. No new AEs or SAEs since the last visit. Thanks, Marcus Lee"""  # noqa: E501

ANSWER_3 = {
    "visit": {"visit_type": "remote visit", "visit_date": "Oct 2, 2026", "site": "site 552",
              "monitor": "Marcus Lee", "pi": None, "screened": None, "enrolled": None},
    "findings": [
        {"issue": "DATA_ENTRY_BACKLOG", "status": "no_issue", "severity": "minor",
         "evidence": "eCRF pages are current", "explanation": "Data entry is up to date."},
        {"issue": "SDV_BACKLOG", "status": "no_issue", "severity": "minor",
         "evidence": "SDV is up to date through Visit 6", "explanation": "SDV is current."},
        {"issue": "PROTOCOL_DEVIATION", "status": "active", "severity": "minor",
         "evidence": "One subject's Week 12 visit fell two days outside the window because of a holiday closure",
         "explanation": "A single out-of-window visit, already logged; minor."},
        {"issue": "SAE_REPORTING", "status": "no_issue", "severity": "minor",
         "evidence": "No new AEs or SAEs since the last visit.", "explanation": "No safety events to report."},
    ],
    "actions": [],
    "summary": "Routine remote visit with data and SDV current. One minor out-of-window visit was logged as a "
               "deviation; nothing else is open.",
}

for _answer in (ANSWER_1, ANSWER_2, ANSWER_3):
    for _f in _answer["findings"]:
        for _k, _v in (("repeat", False), ("subjects_affected", None), ("site_wide", False),
                       ("escalation_evidence", "")):
            _f.setdefault(_k, _v)


def _dump(answer: dict) -> str:
    return json.dumps({k: answer[k] for k in EXTRACT_SCHEMA["propertyOrdering"]})


SYSTEM = f"""You read clinical trial site monitoring visit notes for a clinical research associate (CRA) and turn
each note into structured data.

Issue types:
{ISSUE_LIST}

{RUBRIC_TEXT}

Rules:
- List every topic from the issue types that the note mentions, including topics mentioned only to confirm they
  are fine (status "no_issue") and problems fixed and verified during the visit ("resolved_on_site").
- "evidence" must be copied word for word from the note: the shortest sentence or clause that shows it.
- Visit details (visit_type, visit_date, site, monitor, pi, screened, enrolled) must be copied exactly as written
  in the note, or null if the note does not state them. monitor and pi are the person's name only, without
  credentials (RN, PhD, MD). screened and enrolled are the numbers only. Never use a date that is not the visit date
  (for example a hospitalization or prescription date) as visit_date.
- Every JSON value is plain data copied from the note or written as instructed. Never put notes to yourself,
  corrections or reasoning inside a value.
- "actions" are open follow-ups still to be done. owner, action and due are copied word for word from the note
  (due may be null). Completed tasks, things already done during the visit, and general reminders are not actions.
- Give each finding its own severity from the rubric, then fill the escalation fields. Do not raise the severity
  yourself for repeats or many subjects; SCOPE does that.
- Do not invent anything. If the note does not say it, leave it out.
- "summary": two sentences a study manager can read in ten seconds.

Example note:
\"\"\"{EXAMPLE_1}\"\"\"
Answer: {_dump(ANSWER_1)}

Example note:
\"\"\"{EXAMPLE_2}\"\"\"
Answer: {_dump(ANSWER_2)}

Example note:
\"\"\"{EXAMPLE_3}\"\"\"
Answer: {_dump(ANSWER_3)}"""

ENGINE_REV = "6.0"  # bump when the engine's behaviour changes, so cached answers are not reused

# visit details that may be taken from a labelled header line when the model leaves them out
HEADER_FALLBACK = {"VISIT_TYPE", "VISIT_DATE", "SITE"}

# test notes that appear in the prompt as worked examples (none: the examples above are written for the prompt)
FEW_SHOT_IDS: set[str] = set()


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


_PIECES = re.compile(r"\s+(?:-|–|—|→|->|=>)\s+|\s*[;|\n]\s*|\s+\(|\)\s*")


def _locate(text: str, value, start: int = 0) -> tuple[int, int] | None:
    """Find a value from the model in the note. If the whole value is not there (the model added extra words),
    accept the first clean piece of it that is."""
    loc = _find(text, value, start) or _find(text, value)
    if loc or not value:
        return loc
    for piece in _PIECES.split(str(value)):
        if len(piece.strip(" .,:")) >= 3:
            loc = _find(text, piece, start) or _find(text, piece)
            if loc:
                return loc
    return None


COUNT_CUES = {"screened": r"screen", "enrolled": r"enrol|randomi[sz]|consented"}


def _count(text: str, value, key: str) -> tuple[int, int] | None:
    """Find a subject count as a whole number, preferring the one next to 'screened' / 'enrolled' words."""
    num = str(value or "").strip()
    if not num:
        return None
    hits = [m.span() for m in re.finditer(rf"(?<![\w-]){re.escape(num)}(?![\w-])", text)]
    for cue in re.finditer(COUNT_CUES[key], text, re.I):
        near = [h for h in hits if abs(h[0] - cue.start()) <= 40]
        if near:
            return min(near, key=lambda h: abs(h[0] - cue.start()))
    return None


def _short(value, n: int = 60) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= n else value[: n - 1] + "…"


UNREADABLE = ("Gemini did not return a usable reading of this note (empty or garbled answer), so SCOPE has not "
              "scored it. Please try again.")
_RAMBLE = re.compile(r"\?|\bwait\b|let'?s (?:check|see)|\bhmm+\b|\bactually\b|\bI think\b", re.I)
_FOREIGN = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff]")


class Unreadable(LLMError):
    """Every attempt gave an empty or garbled answer. Keeps the last answer so it can be inspected."""

    def __init__(self, reason: str, raw: str, model: str | None):
        super().__init__(f"{UNREADABLE} (last problem: {reason}; model: {model or 'unknown'})")
        self.reason, self.raw, self.model = reason, raw, model


def output_problem(data, text: str) -> str | None:
    """Why an LLM answer can't be trusted as a whole (None when it looks sound)."""
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        return "no findings list"
    if not data["findings"] and not str(data.get("summary") or "").strip():
        return "empty answer"
    values = [v for v in (data.get("visit") or {}).values() if isinstance(v, str)]
    values += [str(f.get("evidence", "")) for f in data["findings"] if isinstance(f, dict)]
    for v in values:
        if len(v) > 600 or (_FOREIGN.search(v) and not _FOREIGN.search(text)):
            return "garbled value"
    for v in (data.get("visit") or {}).values():
        if isinstance(v, str) and (len(v) > 120 or (_RAMBLE.search(v) and _norm(v) not in _norm(text))):
            return "reasoning inside a value"
    return None


# Plain-code safety net. If the note uses one of these words and the LLM returned no finding at all (of any
# status) for the matching topic, SCOPE raises an alert instead of quietly trusting the reading.
TRIPWIRES = [
    ({"SAE_REPORTING"}, "a possible serious adverse event",
     r"\bSAEs?\b|serious adverse|hospitali[sz]|\badmitted\b|\badmission\b|\bE[DR] visit|emergency (?:room|department)"
     r"|life[- ]threatening|\bdied\b|\bdeath\b"),
    ({"CONSENT"}, "a possible consent problem",
     r"before (?:\w+ ){0,3}(?:signed|signing) (?:the )?(?:ICF|consent)|without (?:a )?(?:signed )?(?:ICF|consent)"
     r"|verbal(?:ly)? consent|consented after|consent (?:was )?signed after"),
    ({"DOSING_ERROR", "PROTOCOL_DEVIATION", "IP_ACCOUNTABILITY"}, "a possible dosing error",
     r"double dos|wrong dose|wrong kit|overdos|dosing error|dosed (?:in error|despite)"),
    ({"ELIGIBILITY", "PROTOCOL_DEVIATION"}, "a possible eligibility problem",
     r"\bineligible\b|did not meet (?:the )?(?:inclusion|exclusion|eligibility)|eligibility (?:violation|not met)"
     r"|(?:inclusion|exclusion) criteri(?:on|a) (?:not met|violated)"),
    ({"REG_DOCS", "PROTOCOL_DEVIATION", "CONSENT"}, "a possible IRB approval lapse",
     r"IRB[^.\n]{0,60}\b(?:expired|lapsed)|(?:expired|lapsed)[^.\n]{0,30}IRB"),
]


def safety_alerts(text: str, findings: list[dict]) -> list[str]:
    covered = {f["issue"] for f in findings if f.get("verified")}
    alerts = []
    for codes, what, pattern in TRIPWIRES:
        m = re.search(pattern, text, re.I)
        if m and not covered & codes:
            alerts.append(f"The note mentions {what} (\"{m.group(0)}\") but SCOPE returned no finding about it. "
                          "Read this note yourself before relying on the risk level.")
    return alerts


class LLMParser:
    name = "llm"

    def __init__(self, client: GeminiClient, sleep: float = 0.0):
        self.client = client
        self.sleep = sleep

    def extract(self, text: str) -> dict:
        return self.client.generate_json(SYSTEM, f"Note:\n\"\"\"\n{text}\n\"\"\"\nAnswer:", EXTRACT_SCHEMA)

    attempts = 3  # a garbled or empty answer is asked again, never scored

    def read(self, text: str) -> dict:
        """Ask for a reading; a bad answer is asked again, preferring a different model each time."""
        reason, raw = "", ""
        try:
            for _ in range(self.attempts):
                try:
                    data = self.extract(text)
                    reason, raw = output_problem(data, text), json.dumps(data, ensure_ascii=False, indent=1)
                except BadAnswer as e:
                    reason, raw = "not valid JSON", e.raw
                if not reason:
                    return data
                if self.client.model:
                    self.client.avoid.add(self.client.model)
        finally:
            self.client.avoid.clear()
        raise Unreadable(reason, raw, self.client.model)

    def predict(self, text: str) -> dict:
        data = self.read(text)
        raw_answer = copy.deepcopy(data)  # exactly what the LLM said, before SCOPE adds its checks
        tokens = tokenize(text)
        char_spans: list[tuple[str, int, int]] = []
        problems: list[str] = []

        # visit details: keep only values that are really in the note
        v = data.get("visit") or {}
        date_at = re.search(r"(?:date of visit|visit date|date)\s*:", text, re.I)
        header = {lab: (s0, e0) for lab, s0, e0 in RuleParser()._metadata(text, [])}
        for key, label in (("visit_type", "VISIT_TYPE"), ("visit_date", "VISIT_DATE"), ("site", "SITE"),
                           ("monitor", "MONITOR"), ("pi", "PI"), ("screened", "SCREENED"), ("enrolled", "ENROLLED")):
            start = date_at.end() if (key == "visit_date" and date_at) else 0
            loc = _count(text, v.get(key), key) if key in COUNT_CUES else _locate(text, v.get(key), start)
            if not loc and label in HEADER_FALLBACK:
                loc = header.get(label)  # clearly labelled header line, e.g. "Date: ..." or "Visit Type: ..."
            if loc:
                char_spans.append((label, *loc))
            elif v.get(key):
                problems.append(f"'{_short(v.get(key))}' ({key.replace('_', ' ')}) is not in the note")

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
            try:
                f["subjects_affected"] = int(f.get("subjects_affected")) if f.get("subjects_affected") else None
            except (TypeError, ValueError):
                f["subjects_affected"] = None
            claims = bool(f.get("repeat") or f.get("site_wide") or (f["subjects_affected"] or 0) >= ESCALATION_SUBJECTS)
            esc = str(f.get("escalation_evidence") or "")
            # escalation only counts when the note really says it (in the escalation quote or the evidence itself)
            f["escalation_ok"] = claims and bool(esc.strip() and verify_quote(esc, text)) or (
                claims and not esc.strip() and f["verified"])
            if claims and not f["escalation_ok"]:
                problems.append(f"'{f['display']}' was reported as a repeat or widespread problem, but the quote "
                                "for that is not in the note, so it was not raised")
            f["final_severity"], f["escalated_by"] = final_severity(f)
            findings.append(f)
        unverified = [f for f in findings if not f["verified"]]
        if unverified:
            problems.append(f"{len(unverified)} finding(s) quote text that is not in the note and were ignored")
        scored = score_findings(findings)
        alerts = safety_alerts(text, findings)

        # action items: the action text must be in the note; owner and due are looked for in the same sentence
        bounds = sentences(text)
        action_items = []
        for a in data.get("actions") or []:
            loc = _find(text, a.get("action"))
            if not loc:
                problems.append(f"action '{_short(a.get('action'))}' is not in the note")
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
            "summary": data.get("summary", ""), "review_reasons": alerts + problems, "alerts": alerts,
            "backend": self.name,
            "model": self.client.model, "llm_output": raw_answer,
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


__all__ = ["LLMParser", "LLMError", "Unreadable", "SYSTEM", "FEW_SHOT_IDS"]
