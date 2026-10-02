"""Rule-based baseline: regular expressions, a keyword lexicon and NegEx-style negation.

This is how visit-note extraction is usually started: hand-written patterns for
metadata, keyword lists for each issue type, a list of negation cues, and a
points table for risk. It needs no model or GPU, so it is (a) the baseline the
BERT model is measured against and (b) the fallback in the app when no
checkpoint is available. It returns the same structure as the BERT parser.
"""

from __future__ import annotations

import re

from scope.data.templates import VISIT_TYPE_SURFACES
from scope.schema import ISSUE_CODES, SEVERITY_POINTS, risk_from_points
from scope.text import bio_to_spans, char_spans_to_bio, sentences, tokenize

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}-{_MONTH}-\d{{2,4}}|\d{{1,2}}{_MONTH}\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|{_MONTH}\.? \d{{1,2}},? \d{{4}}|\d{{1,2}} {_MONTH} \d{{4}}|\d{{1,2}}\.\d{{1,2}}\.\d{{4}})\b",
    re.I,
)
_SURFACES = sorted({s for forms in VISIT_TYPE_SURFACES.values() for s in forms}
                   | {"monitoring visit", "interim visit", "for-cause visit", "remote"}, key=len, reverse=True)
VISIT_TYPE_RE = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in _SURFACES) + r")\b", re.I)
SITE_RE = re.compile(r"\bsite\s*#?\s*\d{2,4}\b", re.I)
_NAME = r"(?:Dr\. )?(?:[A-Z]\. [A-Z][a-z]+|[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+)"
MONITOR_RES = [
    re.compile(rf"\b(?:CRA|Monitor)\s*:?\s*(?P<name>{_NAME})(?!\s+(?:to|will|agreed|owns))"),
    re.compile(rf"conducted by (?P<name>{_NAME})"),
    re.compile(rf"^(?P<name>{_NAME}) completed the", re.M),
    re.compile(rf"(?:Thanks|Best|Cheers|Regards),?\s*\n?\s*(?P<name>{_NAME})\s*$"),
]
PI_RES = [
    re.compile(rf"\bPI\s*:?\s*(?P<name>{_NAME})(?!\s+(?:to|will|agreed|owns))"),
    re.compile(rf"[Tt]he PI, (?P<name>{_NAME}),"),
    re.compile(r"(?P<name>Dr\. [A-Z][a-z]+(?: [A-Z][a-z]+)?)"),
]
_NUM = r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
SCREENED_RES = [re.compile(rf"\bScr\s+(?P<n>{_NUM})\b", re.I),
                re.compile(rf"\b(?P<n>{_NUM})\s+(?:subjects\s+)?(?:have been\s+)?screened\b", re.I),
                re.compile(rf"\bof (?P<n>{_NUM}) screened\b", re.I),
                re.compile(rf"\bout of (?P<n>{_NUM}) screened\b", re.I)]
ENROLLED_RES = [re.compile(rf"\bRand\s+(?P<n>{_NUM})\b", re.I),
                re.compile(rf"\b(?P<n>{_NUM})\s+(?:subjects\s+)?(?:randomi[sz]ed|enrolled|in the study)\b", re.I),
                re.compile(rf"\brandomi[sz]ed (?P<n>{_NUM}) subjects\b", re.I)]

_OWNER = (r"(?:[Tt]he CRC|CRC|New CRC|Site|site|Study coordinator|Regulatory coordinator|Reg coordinator|Coordinator|"
          r"Sub-I(?: [A-Z][a-z]+)?|PI|Dr\. [A-Z][a-z]+|Pharmacist|Pharmacy|CRA(?: [A-Z][a-z]+)?)")
_DUE_TAIL = r"(?:\s+(?:by|before|no later than|within)\s+(?P<due>[^()]+?)|\s*\(due\s+(?P<due2>[^)]+)\))?"
ACTION_RES = [
    re.compile(rf"^\s*(?:Action:\s*|F/u:\s*)?(?P<owner>{_OWNER})\s+(?:to|will|agreed to)\s+(?P<action>.+?)"
               rf"{_DUE_TAIL}\s*\.?\s*$"),
    re.compile(rf"^\s*-\s*(?P<action>.+?)\s+-\s+(?P<owner>{_OWNER})\s+-\s+(?P<due>.+?)\s*$"),
    re.compile(rf"^\s*AI:\s*(?P<action>.+?)\s*\((?P<owner>{_OWNER}),\s*(?P<due>[^)]+)\)\s*$"),
    re.compile(rf"Requested that (?P<owner>{_OWNER}) (?P<action>.+?) by (?P<due>[^.]+)"),
    re.compile(rf"^\s*(?P<owner>{_OWNER}) owns: (?P<action>.+?), target (?P<due>.+?)\.?\s*$"),
]
ACTION_CUE_RE = re.compile(rf"^\s*(?:Action:|AI:|F/u:|-\s*\w.* - {_OWNER} - |{_OWNER}\s+(?:to|will|agreed|owns)\b)"
                           r"|Requested that")

ISSUE_KEYWORDS = {
    "DATA_ENTRY_BACKLOG": r"data entry|\be?crf\b|\bpages\b|edc entry|not entered",
    "QUERY_AGING": r"quer(?:y|ies)",
    "SDV_BACKLOG": r"\bsdv\b|source (?:data )?verification|source documents|source access|emr",
    "SAE_REPORTING": r"\bsaes?\b|serious adverse|hospitali[sz]|unreported",
    "CONSENT": r"\bicfs?\b|consent",
    "PROTOCOL_DEVIATION": r"deviation|\bpds?\b|out[- ]of[- ]window|outside the window|wrong dose|dosed|ineligible|"
                          r"inclusion criterion|eligibility",
    "IP_ACCOUNTABILITY": r"accountability|dispens|\bkits?\b|returned ip|unaccounted",
    "TEMP_EXCURSION": r"temperature|\btemp\b|excursion|fridge|freezer|logger",
    "STAFF_TURNOVER": r"\bstaff\b|resigned|turnover|on leave|new (?:crc|coordinator|sub-i|sub-investigator|nurses?)|"
                      r"training|coordinator (?:left|change)",
    "PI_OVERSIGHT": r"\bpi\b[^.]*(?:sign|review|involvement|unavailable|oversight)|oversight",
    "ENROLLMENT_LAG": r"(?:enrol|recruit|screen)[^.]*(?:behind|slow|stalled|target|fail|no new|no subjects|"
                      r"no referrals)|no new subjects|enrolled no",
    "REG_DOCS": r"\bisf\b|delegation log not signed|regulatory binder|reg docs|\bbinder\b|essential documents|"
                r"irb approval|licen[cs]es?|"
                r"\bcvs?\b|1572|lab normal",
}
_ISSUE_RES = {c: re.compile(p, re.I) for c, p in ISSUE_KEYWORDS.items()}
NEGATION_RE = re.compile(
    r"^\W*(?:no|none|nothing|0)\b|\bno (?:new|issues|findings|discrepancies|excursions|deviations|concerns|"
    r"consent issues|open|unreported|staff changes|outstanding)\b|\bno \w+ concerns|nothing new|none outstanding|"
    r"\b0 open\b|resolved|cleared|have been closed|and closed|been completed|completed for all|has completed|"
    r"complete for all|up to date|is current|on track|ahead of|\badequate|acceptable|caught up|addressed|corrected|"
    r"renewed|filled|recovered|released|stable|meeting expectations|\breconciled|complete and|"
    r"signed and dated correctly|all (?:icfs|queries|ecrf)|capa is in place|(?:was|were) received|updated for|"
    r"now all entered|added and verified|documents filed", re.I)
HEADING_RE = re.compile(r"^\s*[A-Z][A-Za-z ,&/]{2,45}$")
CRITICAL_RE = re.compile(
    r"critical|never reported|not reported|before signing|prior to consent|no signed consent|ineligible|wrong dose|"
    r"expired ip|dispensed before|compromised|lapsed|untrained|denied source|inadequate|no data has been entered|"
    r"closure|missing from the pharmacy|death", re.I)
MAJOR_RE = re.compile(
    r"major|significant|well behind|large|repeated|not counted|unaccounted|wrong subject|outside the 24|days after|"
    r"late\b|growing|unresponsive|not provided|unavailable for|not reviewed|has not|not filed|expired|not signed|"
    r"not on the delegation|resigned|turnover|\b\d{2,} (?:pages|queries|open|missing)|over \d+|important|"
    r"not yet trained|not maintained|quarantined|limited|no new subjects|stalled|old consent|not re-consented|"
    r"high", re.I)
RISK_LINE_RE = re.compile(r"risk(?: level| assessment)?\s*[:-]?\s*(low|medium|moderate|high)", re.I)


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    while e > s and text[e - 1] in " .,;:)":
        e -= 1
    while s < e and text[s] in " (-":
        s += 1
    return s, e


class RuleParser:
    name = "rules"

    # ---------------------------------------------------------------- spans
    def _metadata(self, text: str, action_ranges: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
        def in_action(pos: int) -> bool:
            return any(s <= pos < e for s, e in action_ranges)

        out = []
        m = VISIT_TYPE_RE.search(text)
        if m:
            out.append(("VISIT_TYPE", m.start(), m.end()))
        m = SITE_RE.search(text)
        if m:
            out.append(("SITE", m.start(), m.end()))
        blocked = re.compile(r"(?:next|last|previous|due|by|before|since|later than|target|pre-visit|lapsed|failure|"
                             r"from|oldest|aware|on or|excursion|planned for|was on)\W*$", re.I)
        for m in DATE_RE.finditer(text):
            if not in_action(m.start()) and not blocked.search(text[max(0, m.start() - 25):m.start()]):
                out.append(("VISIT_DATE", m.start(), m.end()))
                break
        for label, patterns in (("MONITOR", MONITOR_RES), ("PI", PI_RES)):
            for rx in patterns:
                m = next((m for m in rx.finditer(text) if not in_action(m.start("name"))), None)
                if m:
                    out.append((label, *_strip(text, m.start("name"), m.end("name"))))
                    break
        for label, patterns in (("SCREENED", SCREENED_RES), ("ENROLLED", ENROLLED_RES)):
            for rx in patterns:
                m = rx.search(text)
                if m:
                    out.append((label, m.start("n"), m.end("n")))
                    break
        # drop overlaps (keep the earliest-added span)
        kept: list[tuple[str, int, int]] = []
        for sp in out:
            if all(sp[2] <= k[1] or sp[1] >= k[2] for k in kept):
                kept.append(sp)
        return kept

    def _actions(self, text: str) -> tuple[list[tuple[str, int, int]], list[tuple[int, int]]]:
        spans, ranges = [], []
        for s, e in sentences(text):
            sent = text[s:e]
            for rx in ACTION_RES:
                m = rx.search(sent)
                if not m:
                    continue
                for group, label in (("owner", "OWNER"), ("action", "ACTION"), ("due", "DUE"), ("due2", "DUE")):
                    if group in rx.groupindex and m.group(group):
                        a, b = _strip(text, s + m.start(group), s + m.end(group))
                        if b > a:
                            spans.append((label, a, b))
                ranges.append((s, e))
                break
        return spans, ranges

    # --------------------------------------------------------------- issues
    def _issues(self, text: str, action_ranges: list[tuple[int, int]]) -> dict[str, str]:
        found: dict[str, str] = {}
        rank = {"minor": 0, "major": 1, "critical": 2}
        for s, e in sentences(text):
            if any(a <= s < b for a, b in action_ranges):
                continue
            sent = text[s:e]
            if HEADING_RE.match(sent) and len(sent.split()) <= 5:
                continue  # section headings like "Informed consent and safety"
            if NEGATION_RE.search(sent):
                continue
            hits = [c for c in ISSUE_CODES if _ISSUE_RES[c].search(sent)]
            if "TEMP_EXCURSION" in hits and "IP_ACCOUNTABILITY" in hits:
                hits.remove("IP_ACCOUNTABILITY")
            if not hits:
                continue
            sev = "critical" if CRITICAL_RE.search(sent) else "major" if MAJOR_RE.search(sent) else "minor"
            for c in hits:
                if c not in found or rank[sev] > rank[found[c]]:
                    found[c] = sev
        return found

    # -------------------------------------------------------------- predict
    def predict(self, text: str) -> dict:
        tokens = tokenize(text)
        action_spans, action_ranges = self._actions(text)
        char_spans = self._metadata(text, action_ranges) + action_spans
        tags = char_spans_to_bio(tokens, char_spans)
        severities = self._issues(text, action_ranges)
        m = RISK_LINE_RE.search(text)
        if m:
            risk = {"moderate": "medium"}.get(m.group(1).lower(), m.group(1).lower())
        else:
            risk = risk_from_points(sum(SEVERITY_POINTS[s] for s in severities.values()))
        issues = [c for c in ISSUE_CODES if c in severities]
        return {
            "tokens": tokens, "tags": tags, "spans": bio_to_spans(tokens, tags, text),
            "risk": risk, "risk_probs": {}, "issues": issues, "issue_probs": {c: 1.0 for c in issues},
            "severities": severities, "backend": self.name,
        }
