"""Turn model output into a structured visit record.

The models decide *what* the note says (risk level, active issues, and which
words are metadata or action items). This module is deterministic: it
normalizes those spans (dates to ISO, counts to integers, visit types to codes),
groups action spans into owner / action / due items, and lists anything it
could not resolve under ``warnings``.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from scope.schema import ISSUE_BY_CODE, VISIT_TYPES
from scope.text import Span, sentences

_DATE_FORMATS = ["%d-%b-%Y", "%d%b%Y", "%m/%d/%Y", "%B %d, %Y", "%Y-%m-%d", "%d %b %Y", "%m/%d/%y", "%d.%m.%Y",
                 "%b %d, %Y", "%b %d %Y", "%B %d %Y", "%d %B %Y", "%d-%B-%Y", "%d-%b-%y"]
_NUMBER_WORDS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                              "nine", "ten", "eleven", "twelve"])}
_VISIT_TYPE_RULES = [
    ("SIV", r"initiation|\bsiv\b"),
    ("COV", r"close[- ]?out|\bcov\b"),
    ("FOR_CAUSE", r"cause|\bfcv\b"),
    ("REMOTE", r"remote|\brmv\b"),
    ("IMV", r"interim|\bimv\b|monitoring|routine|on-site|visit"),
]


def normalize_date(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.strip().rstrip(".,"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_int(text: str | None) -> int | None:
    if not text:
        return None
    t = text.strip().lower()
    if t.isdigit():
        return int(t)
    return _NUMBER_WORDS.get(t)


def normalize_site(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"\d+", text)
    return m.group(0) if m else None


def normalize_visit_type(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    for code, pattern in _VISIT_TYPE_RULES:
        if re.search(pattern, t):
            return code
    return None


def assemble_actions(text: str, spans: list[Span]) -> list[dict]:
    """Group ACTION spans with the OWNER and DUE spans in the same sentence or line."""
    bounds = sentences(text)

    def sent_of(sp: Span) -> int:
        for i, (s, e) in enumerate(bounds):
            if s <= sp.char_start < e:
                return i
        return -1

    by_sent: dict[int, list[Span]] = {}
    for sp in spans:
        if sp.label in ("ACTION", "OWNER", "DUE"):
            by_sent.setdefault(sent_of(sp), []).append(sp)

    items = []
    for _sid, group in sorted(by_sent.items()):
        group.sort(key=lambda s: s.char_start)
        acts = [s for s in group if s.label == "ACTION"]
        for i, a in enumerate(acts):
            # only look between the neighbouring actions, so items in one sentence don't steal each other's slots
            lo = acts[i - 1].char_end if i > 0 else -1
            hi = acts[i + 1].char_start if i + 1 < len(acts) else 10**9
            owners = [s for s in group if s.label == "OWNER" and lo <= s.char_start < hi]
            dues = [s for s in group if s.label == "DUE" and lo <= s.char_start < hi]
            owner = min(owners, key=lambda s: abs(s.char_start - a.char_start), default=None)
            due = min(dues, key=lambda s: abs(s.char_start - a.char_start), default=None)
            items.append({"action": a.text, "owner": owner.text if owner else None,
                          "due": due.text if due else None, "due_date": normalize_date(due.text) if due else None})
    return items


def build_record(text: str, pred: dict) -> dict:
    """``pred`` comes from a parser: spans, risk, risk_probs, issues, issue_probs, backend."""
    spans: list[Span] = pred["spans"]
    warnings: list[str] = []
    first: dict[str, Span] = {}
    for sp in spans:
        first.setdefault(sp.label, sp)

    def txt(label):
        return first[label].text if label in first else None

    vt_code = normalize_visit_type(txt("VISIT_TYPE"))
    visit_date = normalize_date(txt("VISIT_DATE"))
    visit = {
        "visit_type": {"code": vt_code, "name": VISIT_TYPES.get(vt_code), "text": txt("VISIT_TYPE")},
        "visit_date": {"iso": visit_date, "text": txt("VISIT_DATE")},
        "site": {"id": normalize_site(txt("SITE")), "text": txt("SITE")},
        "monitor": txt("MONITOR"),
        "pi": txt("PI"),
        "screened": normalize_int(txt("SCREENED")),
        "enrolled": normalize_int(txt("ENROLLED")),
    }
    if not visit["site"]["id"]:
        warnings.append("No site number found.")
    if not txt("VISIT_DATE"):
        warnings.append("No visit date found.")
    elif not visit_date:
        warnings.append(f"Could not read the visit date '{txt('VISIT_DATE')}'.")
    if not vt_code:
        warnings.append("Visit type not stated.")
    if visit["screened"] is not None and visit["enrolled"] is not None and visit["enrolled"] > visit["screened"]:
        warnings.append("Enrolled count is larger than screened count; check the numbers.")

    actions = pred["action_items"] if "action_items" in pred else assemble_actions(text, spans)
    for a in actions:
        if not a["owner"]:
            warnings.append(f"Action '{a['action']}' has no owner.")
        if visit_date and a["due_date"] and a["due_date"] < visit_date:
            warnings.append(f"Action '{a['action']}' is due before the visit date.")

    issues = [{"code": c, "display": ISSUE_BY_CODE[c].display, "group": ISSUE_BY_CODE[c].group,
               "confidence": round(float(pred.get("issue_probs", {}).get(c, 1.0)), 4),
               "severity": pred.get("severities", {}).get(c)} for c in pred["issues"]]
    risk_probs = pred.get("risk_probs") or {}
    if pred["risk"] == "high" and not actions:
        warnings.append("High-risk visit with no follow-up actions recorded.")
    if pred.get("truncated"):
        warnings.append("The note was longer than the model's input limit; the end of the note was not read.")

    return {
        "visit": visit,
        "risk": {"level": pred["risk"], "confidence": round(float(risk_probs.get(pred["risk"], 0)), 4) or None,
                 "probabilities": {k: round(float(v), 4) for k, v in risk_probs.items()}},
        "issues": issues,
        "actions": actions,
        "spans": [s.to_dict() for s in spans],
        "warnings": warnings,
        "review": {"needed": bool(pred.get("review_reasons")), "reasons": pred.get("review_reasons") or []},
        "findings": pred.get("findings", []),
        "summary": pred.get("summary", ""),
        "points": pred.get("points"),
        "backend": pred.get("backend"),
        "model": pred.get("model"),
        "llm_output": pred.get("llm_output"),
        "alerts": pred.get("alerts", []),
    }


def audit_summary(rec: dict) -> str:
    """A short, structured Markdown summary for reports and the audit trail."""
    v = rec["visit"]
    head = f"**{v['site']['text'] or 'Unknown site'}** - {v['visit_type']['name'] or 'Visit'}"
    head += f" on {v['visit_date']['iso'] or v['visit_date']['text'] or 'unknown date'}"
    lines = [head]
    who = ", ".join(x for x in [f"Monitor: {v['monitor']}" if v["monitor"] else "",
                                 f"PI: {v['pi']}" if v["pi"] else ""] if x)
    if who:
        lines.append(who)
    if v["screened"] is not None or v["enrolled"] is not None:
        lines.append(f"Enrollment: {v['enrolled'] if v['enrolled'] is not None else '?'} enrolled / "
                     f"{v['screened'] if v['screened'] is not None else '?'} screened")
    conf = rec["risk"]["confidence"]
    lines.append(f"**Risk: {rec['risk']['level'].upper()}**" + (f" ({conf:.0%} confidence)" if conf else ""))
    if rec.get("review", {}).get("needed"):
        lines.append("**Needs human review:** " + " ".join(rec["review"]["reasons"]))
    lines.append("")
    if rec.get("summary"):
        lines.append(rec["summary"])
        lines.append("")
    if rec["issues"]:
        lines.append("Active issues:")
        evidence = {f["issue"]: f["evidence"] for f in rec.get("findings", [])
                    if f.get("verified") and f["status"] == "active"}
        for i in rec["issues"]:
            sev = f" - {i['severity']}" if i.get("severity") else ""
            quote = f': "{evidence[i["code"]]}"' if i["code"] in evidence else ""
            lines.append(f"- {i['display']} ({i['group']}{sev}){quote}")
    else:
        lines.append("Active issues: none")
    fixed = [f for f in rec.get("findings", []) if f.get("verified") and f["status"] == "resolved_on_site"]
    if fixed:
        lines.append("")
        lines.append("Resolved during the visit:")
        lines += [f"- {f['display']}: \"{f['evidence']}\"" for f in fixed]
    lines.append("")
    if rec["actions"]:
        lines.append("| # | Action | Owner | Due |")
        lines.append("|---|---|---|---|")
        for n, a in enumerate(rec["actions"], 1):
            due = a["due_date"] or a["due"] or "-"
            lines.append(f"| {n} | {a['action']} | {a['owner'] or '-'} | {due} |")
    else:
        lines.append("Action items: none")
    if rec["warnings"]:
        lines.append("")
        lines += [f"> Check: {w}" for w in rec["warnings"]]
    lines.append("")
    lines.append(f"_Generated by SCOPE ({rec.get('backend') or 'unknown'} parser) on {date.today().isoformat()}._")
    return "\n".join(lines)


def to_row(rec: dict) -> dict:
    """Flat row for CSV export / portfolio tables."""
    v = rec["visit"]
    return {
        "site": v["site"]["id"], "visit_date": v["visit_date"]["iso"], "visit_type": v["visit_type"]["code"],
        "monitor": v["monitor"], "risk": rec["risk"]["level"], "risk_confidence": rec["risk"]["confidence"],
        "issues": "; ".join(i["code"] for i in rec["issues"]), "n_issues": len(rec["issues"]),
        "n_actions": len(rec["actions"]), "screened": v["screened"], "enrolled": v["enrolled"],
        "needs_review": rec.get("review", {}).get("needed", False),
        "safety_alert": bool(rec.get("alerts")),
    }
