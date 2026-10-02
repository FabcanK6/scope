"""Label sets shared by the data generator, the models and the evaluation code.

SCOPE reads one site-visit note and predicts three things at once:

* ``risk``    - one overall site-risk level for the visit (low / medium / high)
* ``issues``  - which of 12 issue types are *active* in the note (multi-label)
* ``spans``   - word-level BIO tags for visit metadata and follow-up action items
"""

from __future__ import annotations

from dataclasses import dataclass

RISK_LEVELS = ["low", "medium", "high"]
RISK2ID = {r: i for i, r in enumerate(RISK_LEVELS)}
ID2RISK = dict(enumerate(RISK_LEVELS))


@dataclass(frozen=True)
class Issue:
    code: str
    group: str
    display: str


ISSUES: list[Issue] = [
    Issue("DATA_ENTRY_BACKLOG", "Data & queries", "Data entry backlog / missing pages"),
    Issue("QUERY_AGING", "Data & queries", "Open or aging queries"),
    Issue("SDV_BACKLOG", "Data & queries", "Source data verification behind"),
    Issue("SAE_REPORTING", "Patient safety & consent", "Late or missing SAE reporting"),
    Issue("CONSENT", "Patient safety & consent", "Informed consent issue"),
    Issue("PROTOCOL_DEVIATION", "Protocol & drug", "Protocol deviation"),
    Issue("IP_ACCOUNTABILITY", "Protocol & drug", "Investigational product accountability"),
    Issue("TEMP_EXCURSION", "Protocol & drug", "IP temperature excursion"),
    Issue("STAFF_TURNOVER", "Site operations", "Staff turnover / training gap"),
    Issue("PI_OVERSIGHT", "Site operations", "PI oversight gap"),
    Issue("ENROLLMENT_LAG", "Site operations", "Enrollment behind target"),
    Issue("REG_DOCS", "Site operations", "Regulatory binder / essential documents"),
]
ISSUE_CODES = [i.code for i in ISSUES]
ISSUE2ID = {c: i for i, c in enumerate(ISSUE_CODES)}
ISSUE_BY_CODE = {i.code: i for i in ISSUES}
ISSUE_GROUPS = list(dict.fromkeys(i.group for i in ISSUES))

# Span types tagged at word level.
METADATA_TYPES = ["VISIT_TYPE", "VISIT_DATE", "SITE", "MONITOR", "PI", "SCREENED", "ENROLLED"]
ACTION_TYPES = ["ACTION", "OWNER", "DUE"]
ENTITY_TYPES = METADATA_TYPES + ACTION_TYPES

BIO_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2LABEL = dict(enumerate(BIO_LABELS))

# Canonical visit-type codes and their display names.
VISIT_TYPES = {
    "SIV": "Site initiation visit",
    "IMV": "Interim monitoring visit",
    "REMOTE": "Remote monitoring visit",
    "COV": "Close-out visit",
    "FOR_CAUSE": "For-cause visit",
}

# Severity points used to derive the risk label in the synthetic data
# (and by the rule baseline): low < 2 <= medium < 6 <= high.
SEVERITY_POINTS = {"minor": 1, "major": 3, "critical": 6}


def risk_from_points(points: int) -> str:
    if points >= 6:
        return "high"
    if points >= 2:
        return "medium"
    return "low"
