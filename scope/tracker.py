"""Site history and open action items across visits, per study and site.

Every visit SCOPE reads for a study is added to that site's history: date, visit type, risk, active findings and
action items. At the next visit to the same site, SCOPE shows what is still open from earlier visits, and notices
when a problem that was active last time is active again (the rubric raises repeat findings one level, but only
when the note says it is a repeat, so SCOPE points it out for the reader to check).

The history lives in the user's own browser (like saved studies): nothing is stored on a server. Structure:
``{study name: {site: [visit, ...]}}``, newest last, at most ``MAX_VISITS`` per site.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re

MAX_VISITS = 50


def site_key(record: dict) -> str | None:
    site = record.get("visit", {}).get("site", {})
    key = site.get("id") or site.get("text")
    return str(key).strip() if key and str(key).strip() else None


def _note_key(note: str) -> str:
    return hashlib.sha256(" ".join(note.lower().split()).encode()).hexdigest()[:16]


def _action_id(visit_key: str, action: str) -> str:
    return hashlib.sha256(f"{visit_key}:{action.lower().strip()}".encode()).hexdigest()[:12]


def visit_entry(record: dict, note: str) -> dict:
    v = record["visit"]
    key = _note_key(note)
    findings = [f for f in record.get("findings", []) if f.get("verified") and f["status"] == "active"]
    return {
        "key": key,
        "date": v["visit_date"]["iso"] or v["visit_date"]["text"] or "",
        "visit_type": v["visit_type"]["code"] or v["visit_type"]["text"] or "",
        "risk": record["risk"]["level"],
        "findings": [{"issue": f["issue"], "display": f["display"], "severity": f.get("final_severity", f["severity"]),
                      "evidence": f["evidence"]} for f in findings],
        "actions": [{"id": _action_id(key, a["action"]), "action": a["action"], "owner": a.get("owner") or "",
                     "due": a.get("due_date") or a.get("due") or "", "done": False}
                    for a in record.get("actions", [])],
        "saved": _dt.date.today().isoformat(),
    }


def _sort_key(entry: dict) -> tuple:
    d = entry.get("date") or ""
    return (0, d) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) else (1, entry.get("saved", ""))


def add_visit(tracker: dict, study: str, site: str, entry: dict) -> bool:
    """Add (or refresh) a visit. Re-reading the same note keeps the done ticks. Returns True if it was new."""
    visits = tracker.setdefault(study, {}).setdefault(site, [])
    for i, old in enumerate(visits):
        if old["key"] == entry["key"]:
            done = {a["id"] for a in old["actions"] if a.get("done")}
            entry = {**entry, "actions": [{**a, "done": a["id"] in done} for a in entry["actions"]]}
            visits[i] = entry
            visits.sort(key=_sort_key)
            return False
    visits.append(entry)
    visits.sort(key=_sort_key)
    del visits[:-MAX_VISITS]
    return True


def earlier(tracker: dict, study: str, site: str, key: str | None = None) -> list[dict]:
    """Visits to this site other than the one being read, oldest first."""
    return [v for v in tracker.get(study, {}).get(site, []) if v["key"] != key]


def open_actions(tracker: dict, study: str, site: str, key: str | None = None) -> list[dict]:
    """Action items from other visits to this site that are not ticked done, with the visit date."""
    return [{**a, "visit_date": v["date"], "visit_key": v["key"]}
            for v in earlier(tracker, study, site, key) for a in v["actions"] if not a.get("done")]


def set_done(tracker: dict, study: str, site: str, action_id: str, done: bool = True) -> bool:
    for v in tracker.get(study, {}).get(site, []):
        for a in v["actions"]:
            if a["id"] == action_id:
                a["done"] = bool(done)
                return True
    return False


def repeat_hints(tracker: dict, study: str, site: str, record: dict, note: str) -> list[dict]:
    """Topics active in this note that were also active at the previous visit to the site, where the note itself
    does not call it a repeat. The rubric raises repeats only on the note's word, so these are for the reader."""
    key = _note_key(note)
    before = earlier(tracker, study, site, key)
    if not before:
        return []
    last = before[-1]
    was = {f["issue"]: f for f in last["findings"]}
    hints = []
    for f in record.get("findings", []):
        if f.get("verified") and f["status"] == "active" and f["issue"] in was and not f.get("repeat"):
            hints.append({"display": f["display"], "last_date": last["date"],
                          "last_severity": was[f["issue"]]["severity"]})
    return hints


def site_summary(tracker: dict, study: str) -> list[dict]:
    """One row per site: visits, last visit, last risk, open actions."""
    rows = []
    for site, visits in sorted(tracker.get(study, {}).items()):
        last = visits[-1] if visits else {}
        rows.append({"site": site, "visits": len(visits), "last_visit": last.get("date", ""),
                     "last_risk": last.get("risk", ""),
                     "open_actions": sum(not a.get("done") for v in visits for a in v["actions"])})
    return rows
