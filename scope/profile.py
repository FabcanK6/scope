"""Study profiles: SCOPE's rubric as editable data instead of code.

A profile holds everything that decides how a visit is scored:

* ``topics``       - the issue types, each with what counts as minor / major / critical (editable, can be hidden,
                     and new study-specific topics can be added)
* ``study_rules``  - plain-language rules from the protocol ("A missed Week 4 PK sample is critical")
* ``escalation``   - how many subjects make a problem "widespread", how far that raises it, whether repeats escalate
* ``thresholds``   - the points needed for medium and high risk
* ``corrections``  - user corrections of earlier results; approved ones are shown to the LLM as examples for
                     similar notes, so SCOPE adapts to the study without retraining
* ``changes``      - a dated log of edits (who changed what), for the audit trail

The default profile is the expert-approved rubric v3.2. Profiles are saved and shared as JSON files.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import re

SEVERITIES = ["minor", "major", "critical"]
SEVERITY_POINTS = {"minor": 1, "major": 3, "critical": 6}

# (code, group, display, minor, major, critical) - rubric v3.2, approved by an experienced clinical research
# professional on 2026-10-03
# (v3.2: PI not assessing AEs is PI oversight; how eligibility questions are answered is PI oversight).
_V31 = [
    ("SAE_REPORTING", "Patient safety & consent", "Late or missing SAE reporting",
     "SAE form detail wrong, corrected",
     "SAE follow-up report overdue; PI causality not documented",
     "SAE unreported, or reported outside 24 hours (stays critical even with a CAPA)"),
    ("AE_REPORTING", "Patient safety & consent", "Adverse event recording",
     "one AE entered late", "AEs missing from EDC or not graded. (The PI not assessing AEs is PI_OVERSIGHT, a "
     "separate problem.)", ""),
    ("CONSENT", "Patient safety & consent", "Informed consent",
     "missing time of signature; initials missing on a page",
     "outdated ICF version used (major even if the subject was re-consented during the visit); re-consent overdue",
     "study procedures before consent; no signed ICF"),
    ("ELIGIBILITY", "Patient safety & consent", "Eligibility",
     "eligibility checklist unsigned but criteria met",
     "eligibility evidence missing from source at randomization",
     "ineligible subject randomized or dosed. Only report eligibility when a subject's eligibility is actually in "
     "doubt or undocumented; how eligibility questions get answered (e.g. by email from an absent PI) is PI_OVERSIGHT"),
    ("SAFETY_REPORTS", "Patient safety & consent", "Safety reports to IRB and PI",
     "IND safety reports filed late in the ISF",
     "safety reports not reviewed by the PI or not sent to the IRB", ""),
    ("UNBLINDING", "Patient safety & consent", "Blinding",
     "", "blinded staff could see unblinded documents, no unblinding", "unplanned unblinding not reported"),
    ("PROTOCOL_DEVIATION", "Protocol & drug", "Protocol deviation",
     "a single out-of-window visit or assessment (including one rescheduled outside its window)",
     "important deviations; safety assessments missed entirely (not just late); deviations not logged", ""),
    ("DOSING_ERROR", "Protocol & drug", "Dosing error",
     "dosing time not recorded",
     "missed doses undocumented; compliance not reconciled (including returned doses not counted)",
     "wrong dose, double dose, or dosed despite a hold criterion"),
    ("IP_ACCOUNTABILITY", "Protocol & drug", "IP accountability",
     "small count difference explained on site",
     "kits unaccounted for; wrong kit dispensed",
     "expired investigational product dispensed"),
    ("TEMP_EXCURSION", "Protocol & drug", "IP storage and temperature",
     "brief excursion with no product impact, reported; or an excursion that was reported with the product "
     "quarantined and not used while the Sponsor decides",
     "excursion not reported; logs not kept",
     "product used after an excursion before Sponsor assessment"),
    ("LAB_SAMPLES", "Protocol & drug", "Lab samples and kits",
     "lab kit supplies running low",
     "samples mishandled or not shipped; central lab results not reviewed", ""),
    ("DATA_ENTRY_BACKLOG", "Data quality", "Data entry backlog",
     "a few pages or one concomitant medication not entered",
     "backlog older than 60 days, or a large volume (about 25 or more pages not entered)", ""),
    ("QUERY_AGING", "Data quality", "Open or aging queries",
     "a few queries open",
     "many queries open more than 60 days, or 10 or more queries older than 30 days", ""),
    ("SDV_BACKLOG", "Data quality", "SDV and source access",
     "SDV slightly behind plan",
     "SDV far behind; the monitor's EMR access lapsed",
     "the site refuses access to source documents"),
    ("SOURCE_DOCS", "Data quality", "Source documentation",
     "corrections not initialed or dated",
     "source missing or contradicts EDC; ALCOA+ failures",
     "falsified or back-dated records"),
    ("STAFF_TURNOVER", "Site operations", "Staff, training and delegation",
     "one CV or GCP certificate expired; training still to be completed before the person starts study work",
     "staff not on the delegation log who are not yet doing study work; turnover with no backup",
     "staff already performing study procedures or running visits without delegation or training"),
    ("PI_OVERSIGHT", "Site operations", "PI oversight",
     "one late sign-off",
     "PI not signing labs or eCRFs (a backlog or a long delay); PI not assessing AEs (causality or grade); PI "
     "unavailable to the team. A routine request for the PI to sign items before the next contact is an action item, "
     "not a finding", ""),
    ("ENROLLMENT_LAG", "Site operations", "Enrollment",
     "slightly behind target", "far behind target", ""),
    ("REG_DOCS", "Site operations", "Regulatory and essential documents",
     "one document misfiled",
     "missing 1572, amendment approval or licenses; a pending IRB approval that blocks screening",
     "enrolling after IRB approval lapsed"),
    ("FACILITY_EQUIPMENT", "Site operations", "Facility and equipment",
     "calibration due soon",
     "equipment out of calibration; lab certification (CLIA/CAP) expired", ""),
    ("PRIOR_ACTIONS", "Site operations", "Follow-up of prior findings",
     "one prior action slightly overdue", "prior actions not done; CAPA not implemented", ""),
    ("SITE_ENGAGEMENT", "Site operations", "Site engagement",
     "slow replies to the monitor", "unresponsive for weeks; repeated visit cancellations", ""),
]

GROUPS = ["Patient safety & consent", "Protocol & drug", "Data quality", "Site operations"]


def default_profile() -> dict:
    return {
        "name": "SCOPE standard",
        "version": "3.2",
        "description": "Severity rubric v3.2, written and approved by an experienced clinical research professional.",
        "topics": [{"code": c, "group": g, "display": d, "minor": mi, "major": ma, "critical": cr, "enabled": True}
                   for c, g, d, mi, ma, cr in _V31],
        "study_rules": [],
        "escalation": {"subjects_threshold": 3, "subjects_max": "major", "repeat": True},
        "thresholds": {"medium": 3, "high": 6},
        "deadlines": [{"topic": "SAE_REPORTING", "amount": 24, "unit": "hours", "severity": "critical",
                       "what": "SAEs reported to the sponsor", "source": "SCOPE standard"}],
        "corrections": [],
        "changes": [],
    }


# ---------------------------------------------------------------------------
# validation and helpers
# ---------------------------------------------------------------------------
class ProfileError(ValueError):
    """A profile file that SCOPE cannot use; the message says what to fix."""


def code_for(display: str) -> str:
    """Code for a new study-specific topic: 'Missed PK samples' -> 'STUDY_MISSED_PK_SAMPLES'."""
    slug = re.sub(r"[^A-Z0-9]+", "_", display.upper()).strip("_")[:40] or "TOPIC"
    return slug if slug.startswith("STUDY_") else f"STUDY_{slug}"


def validate(profile: dict) -> dict:
    """Check a profile (e.g. an uploaded file) and fill in anything missing from the default. Returns a copy."""
    if not isinstance(profile, dict) or not isinstance(profile.get("topics"), list):
        raise ProfileError("This file is not a SCOPE study profile (no topics list).")
    base = default_profile()
    p = {**base, **copy.deepcopy(profile)}
    topics, seen = [], set()
    for t in p["topics"]:
        if not isinstance(t, dict) or not str(t.get("display") or t.get("code") or "").strip():
            continue
        code = str(t.get("code") or code_for(t["display"])).strip().upper()
        code = re.sub(r"[^A-Z0-9_]", "_", code)
        if code in seen:
            raise ProfileError(f"The topic code {code} appears twice.")
        seen.add(code)
        topics.append({"code": code, "group": t.get("group") or "Study-specific",
                       "display": str(t.get("display") or code).strip(),
                       **{s: str(t.get(s) or "").strip() for s in SEVERITIES},
                       "enabled": bool(t.get("enabled", True))})
    if not any(t["enabled"] for t in topics):
        raise ProfileError("At least one topic must be switched on.")
    p["topics"] = topics
    p["study_rules"] = [str(r).strip() for r in p.get("study_rules") or [] if str(r).strip()]
    esc = {**base["escalation"], **(p.get("escalation") or {})}
    esc["subjects_threshold"] = max(2, int(esc["subjects_threshold"]))
    if esc["subjects_max"] not in SEVERITIES:
        raise ProfileError("Escalation 'raise up to' must be minor, major or critical.")
    esc["repeat"] = bool(esc["repeat"])
    p["escalation"] = esc
    th = {**base["thresholds"], **(p.get("thresholds") or {})}
    th = {"medium": int(th["medium"]), "high": int(th["high"])}
    if not 0 < th["medium"] < th["high"]:
        raise ProfileError("Points for medium risk must be above 0 and below the points for high risk.")
    p["thresholds"] = th
    deadlines = []
    for d in p.get("deadlines") or []:
        try:
            amount = float(d["amount"])
        except (KeyError, TypeError, ValueError):
            continue
        if d.get("unit") in ("hours", "calendar_days", "business_days") and amount > 0 and d.get("topic"):
            amount = int(amount) if float(amount).is_integer() else amount
            deadlines.append({"topic": str(d["topic"]), "amount": amount, "unit": d["unit"],
                              "severity": d.get("severity") if d.get("severity") in SEVERITIES else "critical",
                              "what": str(d.get("what") or ""), "source": str(d.get("source") or "")})
    p["deadlines"] = deadlines
    p["corrections"] = [c for c in p.get("corrections") or [] if isinstance(c, dict) and c.get("quote")]
    p["changes"] = list(p.get("changes") or [])
    p["name"] = str(p.get("name") or "Untitled study profile").strip()
    p["version"] = str(p.get("version") or "1")
    return p


def loads(text: str) -> dict:
    try:
        return validate(json.loads(text))
    except json.JSONDecodeError as e:
        raise ProfileError(f"The file is not valid JSON ({e.msg}, line {e.lineno}).") from None


def dumps(profile: dict) -> str:
    return json.dumps(profile, indent=2, ensure_ascii=False)


def fingerprint(profile: dict) -> str:
    """Short hash of everything that affects scoring (not the change log)."""
    keep = {k: profile[k] for k in ("topics", "study_rules", "escalation", "thresholds", "deadlines")}
    keep["corrections"] = [c for c in profile["corrections"] if c.get("approved")]
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def label(profile: dict) -> dict:
    return {"name": profile["name"], "version": profile["version"], "fingerprint": fingerprint(profile)}


def enabled_topics(profile: dict) -> list[dict]:
    return [t for t in profile["topics"] if t["enabled"]]


def deadline_map(profile: dict) -> dict[str, dict]:
    return {d["topic"]: d for d in profile.get("deadlines") or []}


def topic_map(profile: dict) -> dict[str, dict]:
    return {t["code"]: t for t in profile["topics"]}


def log_change(profile: dict, summary: str, who: str = "") -> None:
    profile["changes"].append({"date": _dt.date.today().isoformat(), "by": who, "change": summary})


def bump_version(version: str) -> str:
    """'3.1' -> '3.2', '1' -> '2', 'v2-draft' -> 'v2-draft.1'."""
    m = re.match(r"^(.*?)(\d+)$", version)
    return f"{m.group(1)}{int(m.group(2)) + 1}" if m else f"{version}.1"


# ---------------------------------------------------------------------------
# scoring with a profile
# ---------------------------------------------------------------------------
def risk_from_points(points: int, profile: dict) -> str:
    th = profile["thresholds"]
    return "high" if points >= th["high"] else "medium" if points >= th["medium"] else "low"


def final_severity(f: dict, profile: dict) -> tuple[str, list[str]]:
    """Escalation: a problem affecting ``subjects_threshold``+ subjects (or site-wide) is raised one level, up to
    ``subjects_max``; a repeat finding is raised one level (if switched on). Both can apply. Only counted when the
    escalation evidence is really in the note (``escalation_ok``)."""
    esc = profile["escalation"]
    level = SEVERITIES.index(f["severity"])
    reasons = []
    if f.get("escalation_ok"):
        n = f.get("subjects_affected")
        many = isinstance(n, int) and n >= esc["subjects_threshold"]
        if (many or f.get("site_wide")) and level < SEVERITIES.index(esc["subjects_max"]):
            level += 1
            reasons.append(f"{n} subjects affected" if many else "site-wide")
        if esc["repeat"] and f.get("repeat") and level < len(SEVERITIES) - 1:
            level += 1
            reasons.append("repeat finding")
    return SEVERITIES[level], reasons


def score(findings: list[dict], profile: dict) -> dict:
    """Points from verified, active findings (worst final severity per topic) and the risk level."""
    worst: dict[str, str] = {}
    for f in findings:
        if f["status"] != "active" or not f.get("verified"):
            continue
        sev = f.get("final_severity", f["severity"])
        if f["issue"] not in worst or SEVERITY_POINTS[sev] > SEVERITY_POINTS[worst[f["issue"]]]:
            worst[f["issue"]] = sev
    points = sum(SEVERITY_POINTS[s] for s in worst.values())
    return {"risk": risk_from_points(points, profile), "points": points, "active": worst}


# ---------------------------------------------------------------------------
# the rubric as the LLM reads it
# ---------------------------------------------------------------------------
STATUS_TEXT = """Status: "active" = a problem that still exists after the visit.
"resolved_on_site" = it was corrected and verified during the visit.
"no_issue" = the topic is mentioned only to confirm it is fine.
Critical findings stay "active" even when a CAPA is in place."""


def rubric_text(profile: dict) -> str:
    esc = profile["escalation"]
    lines = [f"Severity rubric: {profile['name']} (version {profile['version']}).",
             "Rate each finding on its own facts; repeats and spread are recorded separately (see the escalation "
             "fields) and SCOPE applies those itself."]
    for t in enabled_topics(profile):
        parts = [f"{s}: {t[s]}" for s in SEVERITIES if t[s]]
        lines.append(f"- {t['code']} ({t['display']}). " + ". ".join(parts) + ("." if parts else ""))
    lines.append(STATUS_TEXT)
    repeat = ('"repeat" = true only when the note says this problem was also found at an earlier visit or an earlier '
              "action on it is still open.") if esc["repeat"] else '"repeat" = always false for this study.'
    lines.append(f"Escalation fields (facts only, SCOPE does the scoring): {repeat} \"subjects_affected\" = how many "
                 "subjects the note says the problem affects (null if not stated). \"site_wide\" = true when the note "
                 "describes it as site-wide or systemic. \"escalation_evidence\" = the words from the note that show "
                 "the repeat or the spread (\"\" if neither).")
    if profile.get("deadlines"):
        lines.append("Reporting deadlines in this study (SCOPE checks the timing itself from the dates you copy):")
        for d in profile["deadlines"]:
            unit = d["unit"].replace("_", " ")
            what = d["what"] or "report"
            lines.append(f"- {d['topic']}: {what} within {d['amount']:g} {unit}; late = {d['severity']}.")
        lines.append('For findings on these topics, copy the date the clock starts (e.g. when the site became aware) '
                     'into "clock_start" and the date of the report into "reported_on", word for word from the note '
                     '(null if the note does not say).')
    if profile["study_rules"]:
        lines.append("Study-specific rules (these come from the protocol and override the rubric above):")
        lines += [f"- {r}" for r in profile["study_rules"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# corrections: user feedback that SCOPE learns from
# ---------------------------------------------------------------------------
def add_correction(profile: dict, *, note: str, issue: str, status: str, severity: str | None, quote: str,
                   reason: str, scope_said: str = "", cra_risk: str | None = None, who: str = "") -> dict:
    """Record a user correction. It is used only after the study lead approves it (``approved``)."""
    c = {"date": _dt.date.today().isoformat(), "by": who, "issue": issue, "status": status,
         "severity": severity if status == "active" else None, "quote": quote.strip(), "reason": reason.strip(),
         "scope_said": scope_said, "cra_risk": cra_risk, "note": note, "approved": False}
    profile["corrections"].append(c)
    return c


def relevant_corrections(profile: dict, note: str, k: int = 6) -> list[dict]:
    """The approved corrections most similar to this note (TF-IDF over the corrected note and quote)."""
    approved = [c for c in profile["corrections"] if c.get("approved")]
    if len(approved) <= k:
        return approved
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [f"{c['quote']} {c.get('note', '')}" for c in approved]
    vec = TfidfVectorizer(stop_words="english").fit(docs + [note])
    sims = cosine_similarity(vec.transform([note]), vec.transform(docs))[0]
    order = sorted(range(len(approved)), key=lambda i: -sims[i])[:k]
    return [approved[i] for i in order]


def corrections_text(profile: dict, note: str) -> str:
    cs = relevant_corrections(profile, note)
    if not cs:
        return ""
    names = topic_map(profile)
    lines = ["Corrections from reviewers on this study's earlier notes "
             "(follow them when a note says something similar):"]
    for c in cs:
        topic = names.get(c["issue"], {}).get("display", c["issue"])
        verdict = (f"active, {c['severity']}" if c["status"] == "active" else
                   "fixed during the visit" if c["status"] == "resolved_on_site" else "not a problem")
        why = f" Reason: {c['reason']}" if c.get("reason") else ""
        lines.append(f"- When a note says \"{c['quote']}\": {c['issue']} ({topic}) is {verdict}.{why}")
    return "\n".join(lines)
