"""Visit report draft: rough notes in, a structured monitoring visit report out.

The report is written from the note and SCOPE's verified reading only (findings with their quotes, actions, visit
details). Anything the note does not say is left as a [placeholder] for the author, never invented. It is a draft
for a person to finish and sign, not a report SCOPE files on its own.
"""

from __future__ import annotations

import json
import re

REPORT_SYSTEM = """You turn rough clinical trial site visit notes into a clear, professional monitoring visit report
draft. Use ONLY facts from the note and from the verified reading given to you. Never add a finding, number, date,
name or outcome that is not there. Where a usual report detail is missing, write a placeholder in square brackets,
for example [Attendees] or [Next visit date].

Write plain Markdown with these sections, in this order:
# Monitoring Visit Report
A two-column table of visit details: Site, Visit date, Visit type, Visit by, Principal Investigator, Subjects
screened / enrolled (placeholders where missing).
## Summary
Three or four sentences: what was reviewed, the overall risk level given, and the most important open problems.
## Findings requiring action
One numbered item per active finding: a bold title (topic and severity), what was found (paraphrase the quoted
evidence, keep subject numbers and dates exactly), and what is needed. Say "None" when there are none.
## Resolved during the visit
Bullets, or "None".
## Areas reviewed with no issues
Bullets naming each area confirmed fine, or "None noted".
## Action items
A table with columns Action, Owner, Due (placeholders where missing).
## Next visit
One line, from the note if it says, otherwise [Next visit date and focus].

Keep the author's meaning; fix spelling and expand shorthand (e.g. "subj" -> "Subject", "ICF" stays ICF).
Neutral, factual tone. No sign-off block."""

_SECTIONS = ["# Monitoring Visit Report", "## Summary", "## Findings requiring action", "## Resolved during the visit",
             "## Areas reviewed with no issues", "## Action items", "## Next visit"]


def report_facts(record: dict) -> dict:
    v = record["visit"]
    findings = [f for f in record.get("findings", []) if f.get("verified")]
    return {
        "site": v["site"]["text"], "visit_date": v["visit_date"]["iso"] or v["visit_date"]["text"],
        "visit_type": v["visit_type"]["name"] or v["visit_type"]["text"], "visit_by": v.get("monitor"),
        "pi": v.get("pi"), "screened": v.get("screened"), "enrolled": v.get("enrolled"),
        "risk_level": record["risk"]["level"],
        "active_findings": [{"topic": f["display"], "severity": f.get("final_severity", f["severity"]),
                             "evidence": f["evidence"], "why": f.get("explanation", ""),
                             "raised_because": f.get("escalated_by", [])}
                            for f in findings if f["status"] == "active"],
        "resolved_on_site": [{"topic": f["display"], "evidence": f["evidence"]}
                             for f in findings if f["status"] == "resolved_on_site"],
        "checked_fine": [{"topic": f["display"], "evidence": f["evidence"]}
                         for f in findings if f["status"] == "no_issue"],
        "action_items": [{"action": a["action"], "owner": a.get("owner"), "due": a.get("due_date") or a.get("due")}
                         for a in record.get("actions", [])],
    }


def draft_report(client, record: dict, note: str) -> str:
    """A report draft from the note and SCOPE's verified reading."""
    prompt = ("Rough notes from the visit:\n\"\"\"\n" + note.strip() + "\n\"\"\"\n\nSCOPE's verified reading (JSON):\n"
              + json.dumps(report_facts(record), indent=2) + "\n\nWrite the report draft.")
    text = client.generate(REPORT_SYSTEM, prompt).strip()
    text = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", text)  # some models wrap Markdown in a code fence
    return text


def missing_sections(report: str) -> list[str]:
    """Sections the draft left out (shown to the author so nothing is silently missing)."""
    low = report.lower()
    return [s.lstrip("# ") for s in _SECTIONS if s.lstrip("# ").lower() not in low]


def placeholders(report: str) -> list[str]:
    return sorted(set(re.findall(r"\[([^\[\]\n]{2,60})\]", report)))
