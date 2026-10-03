"""Label sets shared by the data generator, the models and the evaluation code.

SCOPE reads one site-visit note and predicts three things at once:

* ``risk``    - one overall site-risk level for the visit (low / medium / high)
* ``issues``  - which issue types are *active* in the note (22 in rubric v3; 12 in the v1 baseline)
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


# Rubric v3 (approved by the CRA, 2026-10-03): 22 issue types. The LLM engine uses all of them.
ISSUES: list[Issue] = [
    Issue("SAE_REPORTING", "Patient safety & consent", "Late or missing SAE reporting"),
    Issue("AE_REPORTING", "Patient safety & consent", "Adverse event recording"),
    Issue("CONSENT", "Patient safety & consent", "Informed consent"),
    Issue("ELIGIBILITY", "Patient safety & consent", "Eligibility"),
    Issue("SAFETY_REPORTS", "Patient safety & consent", "Safety reports to IRB and PI"),
    Issue("UNBLINDING", "Patient safety & consent", "Blinding"),
    Issue("PROTOCOL_DEVIATION", "Protocol & drug", "Protocol deviation"),
    Issue("DOSING_ERROR", "Protocol & drug", "Dosing error"),
    Issue("IP_ACCOUNTABILITY", "Protocol & drug", "IP accountability"),
    Issue("TEMP_EXCURSION", "Protocol & drug", "IP storage and temperature"),
    Issue("LAB_SAMPLES", "Protocol & drug", "Lab samples and kits"),
    Issue("DATA_ENTRY_BACKLOG", "Data quality", "Data entry backlog"),
    Issue("QUERY_AGING", "Data quality", "Open or aging queries"),
    Issue("SDV_BACKLOG", "Data quality", "SDV and source access"),
    Issue("SOURCE_DOCS", "Data quality", "Source documentation"),
    Issue("STAFF_TURNOVER", "Site operations", "Staff, training and delegation"),
    Issue("PI_OVERSIGHT", "Site operations", "PI oversight"),
    Issue("ENROLLMENT_LAG", "Site operations", "Enrollment"),
    Issue("REG_DOCS", "Site operations", "Regulatory and essential documents"),
    Issue("FACILITY_EQUIPMENT", "Site operations", "Facility and equipment"),
    Issue("PRIOR_ACTIONS", "Site operations", "Follow-up of prior findings"),
    Issue("SITE_ENGAGEMENT", "Site operations", "Site engagement"),
]
ISSUE_CODES = [i.code for i in ISSUES]
ISSUE_BY_CODE = {i.code: i for i in ISSUES}
ISSUE_GROUPS = list(dict.fromkeys(i.group for i in ISSUES))

# The 12 issue types of the v1 baseline (fine-tuned BERT, rules, synthetic generator). Frozen: the published
# v1 model's issue head has exactly these outputs, in this order.
V1_ISSUE_CODES = ["DATA_ENTRY_BACKLOG", "QUERY_AGING", "SDV_BACKLOG", "SAE_REPORTING", "CONSENT",
                  "PROTOCOL_DEVIATION", "IP_ACCOUNTABILITY", "TEMP_EXCURSION", "STAFF_TURNOVER", "PI_OVERSIGHT",
                  "ENROLLMENT_LAG", "REG_DOCS"]
ISSUE2ID = {c: i for i, c in enumerate(V1_ISSUE_CODES)}

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

# Severity rubric (v3): each active finding scores minor = 1, major = 3, critical = 6 (worst finding per topic).
# A repeat finding is raised one level; one affecting 3+ subjects (or site-wide) is raised one level, up to major.
#   6+ points = high   -> any critical finding, or two major findings
#   3-5 points = medium -> one major finding, or three or more minor findings
#   0-2 points = low
SEVERITY_POINTS = {"minor": 1, "major": 3, "critical": 6}


def risk_from_points(points: int) -> str:
    if points >= 6:
        return "high"
    if points >= 3:
        return "medium"
    return "low"
