"""Sentence banks for the synthetic visit-note generator.

Template syntax
---------------
``[[LABEL:slot]]``  fills ``slot`` and tags the inserted text with span type ``LABEL``
``{slot}``          fills ``slot`` without a tag

Every list is split into *seen* and *held-out* variants (see ``split_variants``):
training data only ever uses seen variants, and the ``test_unseen`` split only
uses held-out ones, so it measures how the models cope with phrasings they never saw.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Issue sentences. Severity drives the synthetic risk label (minor=1, major=3,
# critical=6 points). "negated" and "resolved" mention the topic without the
# issue being active - the main trap for keyword-based extraction.
# ---------------------------------------------------------------------------
ISSUE_SENTENCES: dict[str, dict[str, list[str]]] = {
    "DATA_ENTRY_BACKLOG": {
        "minor": [
            "A handful of eCRF pages ({n_small} total) were not yet entered at the time of the visit.",
            "{n_small} CRF pages outstanding, mostly from the last visit window.",
            "Minor data entry lag: {n_small} pages pending for {subj}.",
            "data entry slightly behind ({n_small} pages)",
        ],
        "major": [
            "Data entry is significantly behind, with {n_big} pages outstanding more than {days} days after the visit.",
            "{n_big} missing pages across {n} subjects; site has not entered visit data since {other_date}.",
            "Major data entry backlog: {n_big} pages overdue, CRC cites workload.",
            "eCRF completion well behind - {n_big} pages missing, oldest > {days} days",
        ],
        "critical": [
            "No data has been entered for the last {n} subject visits; the backlog now exceeds {n_huge} pages and is "
            "blocking the interim analysis.",
            "Critical: EDC entry stopped after the coordinator left, {n_huge} pages outstanding.",
        ],
        "negated": [
            "Data entry is current; no missing pages.",
            "All eCRF pages entered within the 5-day window.",
            "No outstanding pages at this visit.",
            "eCRF entry up to date",
        ],
        "resolved": [
            "The data entry backlog noted at the last visit has been cleared.",
            "Missing pages from the previous IMV are now all entered.",
        ],
    },
    "QUERY_AGING": {
        "minor": [
            "{n_small} queries open longer than 30 days; site expects to close them this week.",
            "A few queries remain open ({n_small}) and the site is working through them.",
            "{n_small} open queries > 30d",
            "Small number of aging queries ({n_small}) on the AE pages.",
        ],
        "major": [
            "{n_big} queries are open, {n} of them older than 60 days.",
            "Query aging is a concern: {n_big} open, oldest from {other_date}.",
            "{n_big} open queries >30d, site unresponsive to reminders",
            "Outstanding queries keep growing ({n_big} open, up from {n} at the last visit).",
        ],
        "critical": [
            "Over {n_huge} queries are open and several are older than 90 days, putting database lock at risk.",
            "Query backlog is now critical: {n_huge} open with no responses in {days} days.",
        ],
        "negated": [
            "No open queries.",
            "All queries answered and closed before the visit.",
            "Query status: 0 open.",
            "queries - none outstanding",
        ],
        "resolved": [
            "Aged queries from the last visit have been closed.",
            "The {n} queries flagged in the previous report are resolved.",
        ],
    },
    "SDV_BACKLOG": {
        "minor": [
            "SDV slightly behind plan: {pct_hi}% complete for enrolled subjects.",
            "Did not finish SDV for {subj}; will complete at the next visit.",
            "SDV about {pct_hi}% done",
        ],
        "major": [
            "SDV is well behind, with only {pct_lo}% of required pages verified.",
            "Source documents were unavailable for {n} subjects so SDV could not be performed.",
            "Large SDV backlog ({n_big} pages) carried over from the last visit",
            "Unable to complete source verification because EMR access was not provided.",
        ],
        "critical": [
            "Site has denied source access for the second visit in a row; SDV has not been done since {other_date}.",
        ],
        "negated": [
            "SDV completed for all subjects seen since the last visit.",
            "100% SDV done, no discrepancies.",
            "Source verification up to date.",
        ],
        "resolved": [
            "SDV backlog from the last visit has been completed.",
            "EMR access issue resolved and SDV caught up.",
        ],
    },
    "SAE_REPORTING": {
        "minor": [
            "SAE follow-up information for {subj} was submitted {days_small} days later than requested.",
            "Follow-up report for an existing SAE ({subj}) slightly late.",
        ],
        "major": [
            "SAE for {subj} (hospitalization) was reported to the sponsor {days} days after site awareness.",
            "The SAE for {subj} was reported outside the 24-hour window.",
            "late SAE report for {subj}, site aware {other_date}, reported {days} days later",
            "Two SAEs were entered in EDC but the paper SAE forms were sent {days_small} days late.",
        ],
        "critical": [
            "A hospitalization for {subj} was never reported as an SAE; found during source review.",
            "Unreported SAE identified for {subj}: ER admission documented in clinic notes but not reported.",
            "Critical finding: death of {subj} not reported to the sponsor within 24 hours.",
        ],
        "negated": [
            "No new SAEs since the last visit.",
            "No SAEs reported; AE log reviewed with no issues.",
            "Safety: nothing new to report.",
            "No unreported SAEs found during source review.",
        ],
        "resolved": [
            "The late SAE from the previous visit has been documented and a CAPA is in place.",
            "Previously missing SAE follow-up for {subj} was received.",
        ],
    },
    "CONSENT": {
        "minor": [
            "ICF for {subj} is missing the time of signature.",
            "Consent version date was not recorded on the log for {subj}.",
            "Minor ICF documentation error for {subj} (initials missing on page 3).",
        ],
        "major": [
            "{subj} was not re-consented on ICF version {ver} within the required time frame.",
            "{n_small} subjects are still on the old consent version.",
            "Consent form for {subj} signed by a staff member not on the delegation log.",
            "ICF v{ver} not yet signed by {n_small} active subjects",
        ],
        "critical": [
            "{subj} underwent study procedures before signing the informed consent form.",
            "No signed consent on file for {subj}.",
            "Screening labs drawn for {subj} prior to consent.",
        ],
        "negated": [
            "All ICFs reviewed, signed and dated correctly.",
            "Consent process documented for all new subjects with no findings.",
            "No consent issues identified.",
            "ICF review: no findings",
        ],
        "resolved": [
            "Re-consent on version {ver} completed for all active subjects.",
            "The ICF signature issue from the last visit has been corrected via note to file.",
        ],
    },
    "PROTOCOL_DEVIATION": {
        "minor": [
            "{subj} visit {visit_no} occurred {days_small} days outside the window (minor deviation).",
            "One minor deviation: missed questionnaire for {subj}.",
            "1 minor PD logged (out-of-window visit).",
        ],
        "major": [
            "{n_small} important protocol deviations since the last visit, including dosing errors.",
            "{subj} was dosed despite meeting a withholding criterion.",
            "Major deviation: inclusion criterion {crit} not met for {subj} but the subject was randomized.",
            "Repeated out-of-window visits ({n_small} in the last month).",
        ],
        "critical": [
            "{subj} received the wrong dose for {days} days due to a protocol misunderstanding.",
            "Ineligible subject randomized and dosed: {subj}.",
        ],
        "negated": [
            "No new protocol deviations noted.",
            "No PDs since the last visit.",
            "Deviation log reviewed, nothing new.",
            "Protocol compliance acceptable with no deviations.",
        ],
        "resolved": [
            "Deviations from the last visit have been reported to the IRB and closed.",
            "The out-of-window visit for {subj} has been documented and closed.",
        ],
    },
    "IP_ACCOUNTABILITY": {
        "minor": [
            "Small drug accountability discrepancy ({n_small} tablets) for {subj}; site to reconcile.",
            "IP dispensing log missing one entry for {subj}.",
        ],
        "major": [
            "Drug accountability does not reconcile: {n_small} kits unaccounted for.",
            "Returned IP was not counted for {n} subjects.",
            "Kit {kit} dispensed to the wrong subject ({subj}).",
            "IP accountability log incomplete for {n} subjects",
        ],
        "critical": [
            "Expired IP (kit {kit}) was dispensed to {subj}.",
            "{n_small} kits missing from the pharmacy with no documentation.",
        ],
        "negated": [
            "IP accountability complete and reconciled.",
            "Drug accountability reviewed with no discrepancies.",
            "IP: reconciled, no issues.",
        ],
        "resolved": [
            "Kit discrepancy from the last visit has been reconciled.",
            "Missing dispensing entries were added and verified.",
        ],
    },
    "TEMP_EXCURSION": {
        "minor": [
            "Brief temperature excursion ({temp} C for under an hour) in the pharmacy fridge; IP not affected "
            "per sponsor.",
            "Temp log missing readings for {n_small} days.",
        ],
        "major": [
            "Temperature excursion to {temp} C on {other_date}; affected kits quarantined pending sponsor assessment.",
            "Fridge temperature logs not maintained for {days} days.",
            "Excursion not reported to the sponsor until this visit.",
            "temp excursion ({temp} C) in IP fridge, kits quarantined",
        ],
        "critical": [
            "Kits exposed to an excursion of {temp} C were dispensed before sponsor assessment.",
            "Freezer failure on {other_date}; all IP on site compromised.",
        ],
        "negated": [
            "Temperature logs complete with no excursions.",
            "No temperature excursions since the last visit.",
            "Storage conditions acceptable and logs reviewed.",
        ],
        "resolved": [
            "The excursion from {other_date} was assessed by the sponsor and the kits released.",
            "Temperature log gaps from the last visit have been addressed with a new data logger.",
        ],
    },
    "STAFF_TURNOVER": {
        "minor": [
            "New sub-investigator added; training documentation pending.",
            "One coordinator on leave for {n_small} weeks with backup in place.",
        ],
        "major": [
            "Lead CRC resigned last month and the replacement has not completed protocol training.",
            "{n_small} of the study staff are new and not yet trained on the protocol.",
            "High staff turnover: third coordinator change this year.",
            "Staff performing assessments are not on the delegation log.",
        ],
        "critical": [
            "No trained coordinator on site since {other_date}; visits are being run by untrained staff.",
        ],
        "negated": [
            "Staffing stable with no changes to the delegation log.",
            "No staff changes since the last visit.",
            "Training records complete for all staff.",
        ],
        "resolved": [
            "New CRC has completed protocol training and is on the delegation log.",
            "Delegation log updated for the staff changes noted at the last visit.",
        ],
    },
    "PI_OVERSIGHT": {
        "minor": [
            "PI signatures on AE logs are {days_small} days behind.",
            "PI was unavailable during the visit; met with the sub-I instead.",
        ],
        "major": [
            "PI has not reviewed or signed lab reports for {n} subjects.",
            "Limited PI involvement, with eligibility decisions made by the coordinator.",
            "PI has not signed off on {n_big} eCRF casebooks.",
            "PI not reviewing labs in a timely manner",
        ],
        "critical": [
            "PI oversight is inadequate: clinically significant labs for {subj} were not reviewed for {days} days.",
        ],
        "negated": [
            "PI engaged and available; all lab reports signed.",
            "PI oversight adequate.",
            "PI met with the CRA and there are no oversight concerns.",
        ],
        "resolved": [
            "PI has caught up on outstanding signatures.",
            "PI oversight concern from the last visit addressed; the PI now reviews labs weekly.",
        ],
    },
    "ENROLLMENT_LAG": {
        "minor": [
            "Enrollment slightly behind target ({enr_lag} of {target} expected).",
            "Screening slower this month due to holidays.",
        ],
        "major": [
            "Enrollment well behind plan: {enr_lag} randomized vs a target of {target}.",
            "No new subjects screened in {days} days.",
            "High screen failure rate ({pct_lo}%) is limiting enrollment.",
            "recruitment stalled, no referrals this month",
        ],
        "critical": [
            "Site has enrolled no subjects since activation {n_small} months ago; recommend a closure discussion.",
        ],
        "negated": [
            "Enrollment on track.",
            "Recruitment ahead of target.",
            "Enrollment meeting expectations.",
        ],
        "resolved": [
            "Enrollment has recovered after the new referral source was added.",
        ],
    },
    "REG_DOCS": {
        "minor": [
            "ISF missing the updated lab normal ranges.",
            "One CV in the regulatory binder is unsigned.",
        ],
        "major": [
            "IRB approval for the current protocol amendment not filed in the ISF.",
            "Medical licenses for {n_small} investigators expired.",
            "Delegation log not signed by the PI for {n_small} new staff.",
            "Essential documents missing: current IB acknowledgement and updated FDA 1572.",
        ],
        "critical": [
            "Site continued enrolling after IRB approval lapsed on {other_date}.",
        ],
        "negated": [
            "Regulatory binder complete and current.",
            "ISF reviewed and all essential documents filed.",
            "Reg docs: no findings.",
        ],
        "resolved": [
            "Expired licenses have been renewed and filed.",
            "ISF gaps from the last visit have been filled.",
        ],
    },
}

# Follow-up actions per issue (lowercase verb phrases).
ISSUE_ACTIONS: dict[str, list[str]] = {
    "DATA_ENTRY_BACKLOG": ["enter all outstanding eCRF pages", "clear the data entry backlog",
                           "bring data entry within the 5-day window", "complete the missing CRF pages"],
    "QUERY_AGING": ["resolve all queries older than 30 days", "answer the open queries", "close outstanding queries",
                    "respond to the aged queries"],
    "SDV_BACKLOG": ["provide EMR access for source verification", "prepare source documents for SDV",
                    "schedule an extra day to complete SDV", "pull source records for the remaining subjects"],
    "SAE_REPORTING": ["file a note to file for the late SAE report", "report the SAE to the sponsor",
                      "retrain staff on SAE reporting timelines", "submit the SAE follow-up form"],
    "CONSENT": ["re-consent affected subjects on the current ICF", "document the consent deviation",
                "retrain staff on the consent process", "file the corrected consent forms"],
    "PROTOCOL_DEVIATION": ["report the deviations to the IRB", "complete the deviation log",
                           "retrain staff on visit windows", "submit a CAPA for the dosing error"],
    "IP_ACCOUNTABILITY": ["reconcile drug accountability", "return the unused kits to the depot",
                          "update the dispensing log", "count the returned IP"],
    "TEMP_EXCURSION": ["quarantine the affected kits", "send the temperature logs to the sponsor",
                       "replace the fridge data logger", "report the excursion to the sponsor"],
    "STAFF_TURNOVER": ["complete protocol training for the new coordinator", "update the delegation log",
                       "file training records for new staff", "identify a backup coordinator"],
    "PI_OVERSIGHT": ["sign all outstanding lab reports", "review and sign the AE logs",
                     "set up a weekly PI review meeting", "sign the pending casebooks"],
    "ENROLLMENT_LAG": ["submit a recruitment plan", "add a new referral source", "review the pre-screening log",
                       "update the enrollment projections"],
    "REG_DOCS": ["file the current IRB approval in the ISF", "renew expired medical licenses",
                 "update the regulatory binder", "collect signed CVs for new staff"],
}
GENERIC_ACTIONS = ["send the follow-up letter", "schedule the next monitoring visit",
                   "escalate the findings to the study manager", "update the site issue log"]

SITE_OWNERS = ["Site", "CRC", "the CRC", "Study coordinator", "Regulatory coordinator", "Sub-I"]
PI_OWNER = "PI"
PHARMACY_OWNERS = ["Pharmacist", "Pharmacy"]
CRA_OWNERS = ["CRA"]

ACTION_TEMPLATES = [
    "[[OWNER:owner]] to [[ACTION:action]] by [[DUE:due]].",
    "Action: [[OWNER:owner]] to [[ACTION:action]] (due [[DUE:due]]).",
    "- [[ACTION:action_cap]] - [[OWNER:owner]] - [[DUE:due]]",
    "F/u: [[OWNER:owner]] will [[ACTION:action]] before [[DUE:due]].",
    "[[OWNER:owner]] agreed to [[ACTION:action]] no later than [[DUE:due]].",
    "AI: [[ACTION:action_cap]] ([[OWNER:owner]], [[DUE:due]])",
    "[[OWNER:owner]] to [[ACTION:action]].",
    "Requested that [[OWNER:owner]] [[ACTION:action]] by [[DUE:due]].",
    "[[OWNER:owner]] owns: [[ACTION:action]], target [[DUE:due]]",
]
RELATIVE_DUES = ["next visit", "the next visit", "end of week", "Friday", "end of month"]

# ---------------------------------------------------------------------------
# Metadata blocks per note style
# ---------------------------------------------------------------------------
FIELD_HEADERS = [
    "[[VISIT_TYPE:vt]] - [[SITE:site]] - [[VISIT_DATE:date]]",
    "[[SITE:site]] [[VISIT_TYPE:vt]] [[VISIT_DATE:date]]",
    "[[VISIT_DATE:date]] [[VISIT_TYPE:vt]] @ [[SITE:site]]",
    "[[VISIT_TYPE:vt]] [[SITE:site]], [[VISIT_DATE:date]]",
]
FIELD_PEOPLE = [
    "CRA: [[MONITOR:cra]] | PI: [[PI:pi]]",
    "CRA [[MONITOR:cra]]; PI [[PI:pi]]",
    "Monitor: [[MONITOR:cra]]",
    "[[MONITOR:cra]] / [[PI:pi]]",
]
FIELD_ENROLLMENT = [
    "Scr [[SCREENED:ns]] / Rand [[ENROLLED:ne]]",
    "[[SCREENED:ns]] screened, [[ENROLLED:ne]] randomized",
    "Enrollment: [[ENROLLED:ne]] randomized of [[SCREENED:ns]] screened",
    "[[ENROLLED:ne]] enrolled ([[SCREENED:ns]] screened)",
]
REPORT_OPENINGS = [
    "Visit summary: [[VISIT_TYPE:vt]] at [[SITE:site]] on [[VISIT_DATE:date]], conducted by [[MONITOR:cra]]. "
    "PI: [[PI:pi]].",
    "[[VISIT_TYPE:vt_title]] Report\nSite: [[SITE:site]]\nVisit date: [[VISIT_DATE:date]]\nMonitor: [[MONITOR:cra]]\n"
    "Principal Investigator: [[PI:pi]]",
    "This [[VISIT_TYPE:vt]] was conducted at [[SITE:site]] on [[VISIT_DATE:date]] by [[MONITOR:cra]], with "
    "[[PI:pi]] available for the exit interview.",
    "[[MONITOR:cra]] completed the [[VISIT_TYPE:vt]] for [[SITE:site]] on [[VISIT_DATE:date]]. The PI, [[PI:pi]], "
    "attended the wrap-up.",
]
REPORT_ENROLLMENT = [
    "To date [[SCREENED:ns]] subjects have been screened and [[ENROLLED:ne]] randomized.",
    "Enrollment status: [[SCREENED:ns]] screened / [[ENROLLED:ne]] enrolled.",
    "The site has randomized [[ENROLLED:ne]] subjects out of [[SCREENED:ns]] screened.",
    "[[ENROLLED:ne]] of [[SCREENED:ns]] screened subjects are enrolled.",
]
# The e-mail style is never used for training: it only appears in test_unseen.
EMAIL_OPENINGS = [
    "Hi team,\nQuick summary from today's [[VISIT_TYPE:vt]] at [[SITE:site]] ([[VISIT_DATE:date]]), with [[PI:pi]] "
    "joining at the end.",
    "Hello all - notes from the [[VISIT_TYPE:vt]] I did at [[SITE:site]] on [[VISIT_DATE:date]]. Met with "
    "[[PI:pi]] briefly.",
]
EMAIL_ENROLLMENT = [
    "They are at [[ENROLLED:ne]] randomized, [[SCREENED:ns]] screened.",
    "Numbers: [[SCREENED:ns]] screened so far and [[ENROLLED:ne]] in the study.",
]
EMAIL_SIGNOFFS = ["Thanks,\n[[MONITOR:cra]]", "Best,\n[[MONITOR:cra]]"]

DISTRACTORS = [
    "Next visit planned for {future_date}.",
    "Last visit: {other_date}.",
    "Previous IMV was on {other_date}.",
    "Follow-up letter to be sent within 10 business days.",
    "Pre-visit letter sent {other_date}.",
]
RISK_LINES = ["Overall site risk: {risk_word}.", "Risk assessment: {risk_word}", "Site risk level - {risk_word}"]

REPORT_SECTIONS = {
    "Data & queries": ["Data Management", "Data quality", "EDC and queries"],
    "Patient safety & consent": ["Safety and Consent", "Subject safety", "Informed consent and safety"],
    "Protocol & drug": ["Protocol Compliance and IP", "Protocol and investigational product", "IP and deviations"],
    "Site operations": ["Site Operations", "Site staff and documents", "Site management"],
}
NO_FINDINGS = ["No findings.", "Nothing to report.", "No issues noted."]
ACTION_HEADERS = ["Action Items", "Follow-up", "Next steps", "Action items for site"]

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------
VISIT_TYPE_SURFACES = {
    "SIV": ["SIV", "site initiation visit", "initiation visit", "Site Initiation Visit"],
    "IMV": ["IMV", "interim monitoring visit", "routine monitoring visit", "on-site monitoring visit"],
    "REMOTE": ["remote monitoring visit", "RMV", "remote visit", "remote IMV"],
    "COV": ["COV", "close-out visit", "closeout visit", "Close Out Visit"],
    "FOR_CAUSE": ["for-cause visit", "for cause visit", "FCV", "for-cause audit visit"],
}
VISIT_TYPE_TITLES = {"SIV": "Site Initiation Visit", "IMV": "Interim Monitoring Visit",
                     "REMOTE": "Remote Monitoring Visit", "COV": "Close-Out Visit", "FOR_CAUSE": "For-Cause Visit"}
# issues that make sense for each visit type
VISIT_TYPE_ISSUES = {
    "SIV": ["REG_DOCS", "STAFF_TURNOVER", "PI_OVERSIGHT", "TEMP_EXCURSION"],
    "REMOTE": ["DATA_ENTRY_BACKLOG", "QUERY_AGING", "SDV_BACKLOG", "SAE_REPORTING", "CONSENT", "PROTOCOL_DEVIATION",
               "STAFF_TURNOVER", "PI_OVERSIGHT", "ENROLLMENT_LAG", "REG_DOCS"],
}
VISIT_TYPE_WEIGHTS = {"IMV": 0.5, "REMOTE": 0.2, "SIV": 0.08, "COV": 0.1, "FOR_CAUSE": 0.12}

SITE_FORMATS = ["Site {id}", "site {id}", "Site #{id}", "Site {id}"]
CITIES = ["Hamburg", "Lyon", "Austin", "Toronto", "Leeds", "Madrid", "Osaka", "Nairobi", "Lima", "Krakow",
          "Boston", "Denver", "Seoul", "Milan", "Atlanta", "Perth"]

FIRST_NAMES = ["Jane", "Amara", "Luis", "Mei", "Tomasz", "Priya", "Kwame", "Sofia", "Daniel", "Aisha", "Hiro",
               "Grace", "Omar", "Elena", "Samuel", "Nadia", "Chloe", "Ravi", "Fatima", "Marco", "Yuki", "Zainab",
               "Ethan", "Leila"]
LAST_NAMES = ["Okafor", "Alvarez", "Chen", "Novak", "Patel", "Mensah", "Rossi", "Kim", "Haddad", "Tanaka",
              "Brooks", "Nguyen", "Osei", "Fischer", "Moreau", "Silva", "Kowalski", "Adeyemi", "Larsen", "Gupta",
              "Reyes", "Ibrahim", "Walsh", "Sato"]

# date formats: (strftime-like pattern handled in generate.format_date, held_out)
DATE_FORMATS = ["%d-%b-%Y", "%d%b%Y", "%m/%d/%Y", "%B %d, %Y", "%Y-%m-%d", "%d %b %Y", "%m/%d/%y", "%d.%m.%Y"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve"]


def split_variants(items: list, every: int = 4) -> tuple[list, list]:
    """Hold out every ``every``-th variant (index 3, 7, ...) when a list has at least ``every`` items."""
    if len(items) < every:
        return list(items), []
    seen = [x for i, x in enumerate(items) if i % every != every - 1]
    held = [x for i, x in enumerate(items) if i % every == every - 1]
    return seen, held
