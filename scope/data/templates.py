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


# ---------------------------------------------------------------------------
# Formal report style (v2): long sentences, abbreviations spelled out, most
# topics mentioned only to confirm there is no problem, findings often fixed
# during the visit. Same keys as ISSUE_SENTENCES.
# ---------------------------------------------------------------------------
FORMAL_ISSUE_SENTENCES: dict[str, dict[str, list[str]]] = {
    "DATA_ENTRY_BACKLOG": {
        "negated": [
            "All Electronic Case Report Form (eCRF) pages are fully entered and monitored for subjects seen to date.",
            "Data entry is current, with all visits entered into the eCRF within the protocol-required timeframe.",
            "A review of the eCRF completion report showed no outstanding pages.",
            "All eCRF pages for completed visits were entered prior to the visit.",
        ],
        "resolved": [
            "A new medication for {subj} was found in the clinic chart but was missing from the eCRF; the "
            "coordinator added it while I was on site, and the entry was checked against source.",
            "Two eCRF pages for {subj} were missing at the start of the visit; the SC entered them during the "
            "visit and they were verified against source.",
            "The adverse event log for {subj} had not been transcribed to the eCRF; this was corrected on site and "
            "the SC was retrained on timely data entry.",
            "Vital signs from the Week {visit_no} visit for {subj} were missing in the eCRF and were entered and "
            "verified during the visit.",
        ],
        "minor": [
            "Data entry for {subj} is slightly behind, with {n_small} eCRF pages from the Week {visit_no} visit not "
            "yet entered.",
            "The concomitant medications log for {subj} has not been updated in the eCRF since {other_date}; the SC "
            "plans to update it this week.",
        ],
        "major": [
            "Data entry remains significantly delayed, with {n_big} eCRF pages outstanding for more than {days} days.",
            "The site has not entered eCRF data for any visit since {other_date} because of coordinator workload.",
            "A review of the eCRF completion report showed {n_big} missing pages across {n} subjects.",
        ],
        "critical": [
            "No eCRF data has been entered for {n} subject visits, and the backlog of {n_huge} pages is now "
            "jeopardizing the planned interim analysis.",
        ],
    },
    "QUERY_AGING": {
        "negated": [
            "All eCRF pages are query-free.",
            "All open queries were answered and closed prior to the visit.",
            "There are no outstanding data queries for this site.",
            "Query resolution is timely, with no queries open longer than 14 days.",
        ],
        "resolved": [
            "The {n} aged queries noted at the previous visit have been answered and closed.",
            "The SC resolved the remaining open queries during the visit, and the query report now shows zero open "
            "items.",
            "Outstanding queries from the last visit were reviewed with the SC and closed during the visit.",
        ],
        "minor": [
            "{n_small} queries remain open, most of them less than 30 days old.",
            "A small number of queries ({n_small}) on the laboratory pages are awaiting site response.",
        ],
        "major": [
            "There are {n_big} open queries, {n} of which have been outstanding for more than 60 days.",
            "Query responses have been slow; {n_big} queries remain unanswered despite repeated reminders.",
            "The query aging report shows {n_big} open queries, the oldest dating from {other_date}.",
        ],
        "critical": [
            "More than {n_huge} queries are open and several are older than 90 days, which puts the database lock "
            "timeline at risk.",
        ],
    },
    "SDV_BACKLOG": {
        "negated": [
            "Source verification was completed for {subj} and {subj2}.",
            "Source data verification (SDV) was completed for all subjects seen since the last visit with no "
            "discrepancies.",
            "SDV is up to date for all randomized subjects.",
            "Source documents were available and well organized, and SDV was completed as planned.",
        ],
        "resolved": [
            "The SDV backlog noted at the previous visit has been completed.",
            "EMR access, which delayed SDV at the last visit, has been restored and SDV is now current.",
            "Source documents that were unavailable at the last visit were provided, and SDV was completed.",
        ],
        "minor": [
            "SDV could not be completed for {subj} because the Week {visit_no} source notes were still being "
            "finalized; it will be completed at the next visit.",
            "SDV is slightly behind plan, at about {pct_hi}% of required pages.",
        ],
        "major": [
            "Source documents were not available for {n} subjects, so SDV could not be performed.",
            "SDV remains well behind plan, with only {pct_lo}% of required pages verified.",
            "The site has not provided EMR access for the monitor, and SDV could not be performed for any subject.",
        ],
        "critical": [
            "The site declined to provide source documents for the second consecutive visit, and no SDV has been "
            "performed since {other_date}.",
        ],
    },
    "SAE_REPORTING": {
        "negated": [
            "No new Serious Adverse Events (SAEs) have occurred since the last visit.",
            "All adverse events were reviewed against source, and no unreported SAEs were identified.",
            "The SAE log is complete, and all SAEs were reported to the Sponsor within 24 hours.",
            "Safety reporting was reviewed and is compliant with protocol timelines.",
        ],
        "resolved": [
            "The late SAE report noted at the previous visit has been documented with a note to file, and the "
            "Corrective and Preventive Action (CAPA) plan has been completed.",
            "Follow-up information for the SAE reported for {subj} was received and entered in the eCRF.",
            "The SAE narrative for {subj} was checked against the hospital records and is consistent.",
        ],
        "minor": [
            "Follow-up information for the SAE reported for {subj} was sent to the Sponsor {days_small} days later "
            "than requested.",
        ],
        "major": [
            "The site became aware of the hospitalization of {subj} on {other_date} but submitted the Serious Adverse "
            "Event (SAE) report {days} days later, outside the 24-hour requirement.",
            "The SAE for {subj} was reported outside of the protocol-mandated 24-hour reporting window.",
            "This for-cause visit was requested by the Sponsor after an SAE for {subj} was reported late.",
        ],
        "critical": [
            "During source review I identified a hospitalization for {subj} that was never reported as a Serious "
            "Adverse Event (SAE).",
            "The death of {subj} on {other_date} was not reported to the Sponsor within 24 hours.",
        ],
    },
    "CONSENT": {
        "negated": [
            "Consent for {subj} was obtained and documented before any study-specific assessments were performed.",
            "Informed consent documentation was reviewed for all new subjects, and all ICFs are on the current "
            "IRB-approved version.",
            "The consent process for {subj} is well documented in the source notes.",
            "All subjects have been re-consented on the current ICF version.",
        ],
        "resolved": [
            "The missing time of signature on the ICF for {subj} was addressed with a note to file during the visit.",
            "Re-consent on ICF version {ver}, which was outstanding at the last visit, has been completed for all "
            "active subjects.",
            "The ICF for {subj} was missing the subject's initials on one page; this was corrected according to "
            "site procedures during the visit.",
        ],
        "minor": [
            "The ICF for {subj} is missing the time of signature.",
            "The consent log does not record the ICF version signed by {subj}.",
        ],
        "major": [
            "{subj} has not been re-consented on ICF version {ver} within the required timeframe.",
            "The ICF for {subj} was signed by a staff member who is not on the delegation log.",
            "{n_small} active subjects remain on an outdated ICF version.",
        ],
        "critical": [
            "{subj} underwent screening procedures before the informed consent form (ICF) was signed.",
            "No signed informed consent form could be located for {subj}.",
        ],
    },
    "PROTOCOL_DEVIATION": {
        "negated": [
            "No new protocol deviations were identified during this visit.",
            "The deviation log was reviewed, and no new deviations have occurred since the last visit.",
            "All visits for active subjects occurred within the protocol-specified windows.",
            "Eligibility was reviewed for the newly randomized subjects, and all inclusion and exclusion criteria "
            "were met.",
        ],
        "resolved": [
            "The protocol deviations noted at the previous visit have been reported to the IRB and closed.",
            "The out-of-window visit for {subj} from the last visit has been documented and closed.",
            "The deviation for {subj} identified at the previous visit has a completed note to file.",
        ],
        "minor": [
            "The Week {visit_no} assessments for {subj} were performed {days_small} days late, outside the visit "
            "window; this was recorded as a minor deviation.",
            "The Week {visit_no} questionnaire was not completed for {subj}, which was logged as a minor deviation.",
        ],
        "major": [
            "{subj} was dosed despite meeting a protocol-defined dose-hold criterion.",
            "{subj} was randomized although inclusion criterion {crit} was not met.",
            "{n_small} important protocol deviations have occurred since the last visit, including dosing errors.",
        ],
        "critical": [
            "{subj} received an incorrect dose for {days} days because of a misunderstanding of the dosing "
            "instructions.",
            "An ineligible subject ({subj}) was randomized and dosed.",
        ],
    },
    "IP_ACCOUNTABILITY": {
        "negated": [
            "A full drug count was completed, and the quantity on hand agrees with the randomization system records.",
            "IP accountability is complete and reconciled for all subjects.",
            "Dispensing records were reviewed, and all kits were dispensed according to the IRT assignments.",
            "Conducted final IP accountability; all remaining kits were reconciled against the shipment records.",
        ],
        "resolved": [
            "The kit discrepancy noted at the previous visit has been reconciled.",
            "Returned IP for {subj}, which had not been counted, was counted and documented during the visit.",
            "A missing dispensing log entry for {subj} was added and verified during the visit.",
        ],
        "minor": [
            "There is a small accountability discrepancy of {n_small} tablets for {subj}; the pharmacist will "
            "reconcile it.",
            "The dispensing log is missing one entry for {subj}.",
        ],
        "major": [
            "IP accountability does not reconcile: {n_small} kits are unaccounted for.",
            "Kit {kit} was dispensed to the wrong subject ({subj}).",
            "Returned IP was not counted for {n} subjects.",
        ],
        "critical": [
            "Expired investigational product (kit {kit}) was dispensed to {subj}.",
            "{n_small} kits are missing from the pharmacy with no documentation of their disposition.",
        ],
    },
    "TEMP_EXCURSION": {
        "negated": [
            "Data logger downloads for the drug storage room show that storage stayed within range.",
            "Temperature logs for the pharmacy refrigerator are complete, with no excursions since the last visit.",
            "The study drug freezer is continuously monitored and alarmed, as the protocol requires.",
            "IP storage conditions were reviewed and are acceptable.",
        ],
        "resolved": [
            "The temperature excursion reported on {other_date} was assessed by the Sponsor, and the affected kits "
            "were released for use.",
            "Gaps in the temperature log noted at the last visit have been addressed with a new data logger.",
        ],
        "minor": [
            "A brief temperature excursion to {temp} C (under one hour) occurred in the pharmacy refrigerator; the "
            "Sponsor confirmed the IP is not affected.",
            "Temperature readings are missing from the log for {n_small} days.",
        ],
        "major": [
            "The pharmacy refrigerator reached {temp} C on {other_date}; the affected kits were quarantined pending "
            "Sponsor assessment.",
            "Temperature logs for the IP refrigerator were not maintained for {days} days.",
            "A temperature excursion was not reported to the Sponsor until this visit.",
        ],
        "critical": [
            "Kits exposed to a temperature excursion of {temp} C were dispensed before the Sponsor assessment was "
            "completed.",
        ],
    },
    "STAFF_TURNOVER": {
        "negated": [
            "The newly hired research nurse's training file and delegation log entry were reviewed and found complete.",
            "There have been no staff changes since the last visit.",
            "Training records are complete for all staff on the delegation log.",
            "Delivered protocol training to the investigator team and {n_small} research nurses.",
        ],
        "resolved": [
            "The new Study Coordinator has completed protocol training and has been added to the delegation log.",
            "The delegation log was updated during the visit to reflect the staff changes noted previously.",
        ],
        "minor": [
            "A newly added Sub-Investigator has not yet completed protocol training documentation.",
            "The primary SC is on leave for {n_small} weeks; a trained back-up is covering visits.",
        ],
        "major": [
            "The lead SC resigned last month, and the replacement has not completed protocol training.",
            "Staff performing study assessments are not listed on the delegation log.",
            "This is the third coordinator change at the site this year.",
        ],
        "critical": [
            "There has been no trained coordinator at the site since {other_date}, and study visits are being "
            "conducted by untrained staff.",
        ],
    },
    "PI_OVERSIGHT": {
        "negated": [
            "All casebooks completed to date have been signed by the investigator.",
            "The Principal Investigator (PI) was available and engaged throughout the visit.",
            "All laboratory reports were reviewed and signed by the PI in a timely manner.",
            "Financial disclosure forms are signed and on file for all investigators.",
        ],
        "resolved": [
            "The PI has caught up on the outstanding laboratory report signatures noted at the last visit.",
            "The PI oversight concern from the previous visit has been addressed; the PI now reviews laboratory "
            "results weekly.",
        ],
        "minor": [
            "PI signatures on the adverse event logs are {days_small} days behind.",
            "The PI was unavailable during the visit; findings were reviewed with the Sub-Investigator.",
        ],
        "major": [
            "The PI has not reviewed or signed laboratory reports for {n} subjects.",
            "Eligibility decisions are being made by the coordinator without documented PI review.",
            "The PI has not signed off on {n_big} eCRF casebooks.",
        ],
        "critical": [
            "Clinically significant laboratory results for {subj} were not reviewed by the PI for {days} days.",
        ],
    },
    "ENROLLMENT_LAG": {
        "negated": [
            "Enrollment is on track with the site's recruitment plan.",
            "Recruitment is ahead of target.",
            "Reviewed the subject recruitment strategy with the PI; screening volume is meeting expectations.",
        ],
        "resolved": [
            "Enrollment has recovered since the new referral source was added.",
        ],
        "minor": [
            "Enrollment is slightly behind target, with {enr_lag} subjects randomized against {target} expected.",
            "Screening slowed this month because of the holiday period.",
        ],
        "major": [
            "Enrollment remains well behind plan, with {enr_lag} subjects randomized against a target of {target}.",
            "No new subjects have been screened in {days} days.",
            "A high screen failure rate ({pct_lo}%) is limiting enrollment.",
        ],
        "critical": [
            "The site has not enrolled any subjects since activation {n_small} months ago, and a site closure "
            "discussion is recommended.",
        ],
    },
    "REG_DOCS": {
        "negated": [
            "The regulatory binder was reviewed, and all essential documents are current.",
            "IRB correspondence and safety letters are filed in date order in the Investigator Site File (ISF).",
            "Essential documents are complete and current.",
            "Emergency equipment in the treatment room was checked and is in date.",
        ],
        "resolved": [
            "The expired medical licenses noted at the last visit have been renewed and filed.",
            "The missing documents in the ISF noted previously have been filed.",
        ],
        "minor": [
            "The ISF is missing the updated laboratory normal ranges.",
            "Screening is on hold until IRB approval of the revised recruitment materials is received.",
            "One CV in the regulatory binder is unsigned.",
        ],
        "major": [
            "IRB approval of the current protocol amendment is not filed in the ISF.",
            "Medical licenses for {n_small} investigators have expired.",
            "The updated FDA Form 1572 has not been filed.",
        ],
        "critical": [
            "The site continued to enroll subjects after IRB approval lapsed on {other_date}.",
        ],
    },
}

# Sentences that mention study topics but are neither findings nor open actions.
FORMAL_NEUTRAL = [
    "The visit started at {hour}:30 with a short meeting with the coordinator.",
    "Walked through the pharmacy and the procedure rooms with the site team.",
    "Record retention obligations were discussed with the investigator.",
    "The pharmacy team was walked through the drug preparation instructions.",
    "Training focused on the inclusion and exclusion criteria and the dosing schedule.",
    "Central lab supplies were checked, and a resupply was requested.",
    "Reviewed the upcoming visit schedule for the active cohort with the SC.",
    "{subj} completed the Week {visit_no} visit on {other_date}.",
    "The PI was notified of all findings at the end of the visit.",
    "The next monitoring visit is tentatively planned for {future_date}.",
    "Unused drug supplies were prepared for return to the depot.",
    "Met with the PI to review the overall status of the study.",
]
FORMAL_ACTION_TEMPLATES = [
    "Follow-up: [[OWNER:owner]] will [[ACTION:action]] by [[DUE:due]].",
    "Pending Items: [[OWNER:owner]] to [[ACTION:action]] by [[DUE:due]].",
    "[[OWNER:owner]] agreed to [[ACTION:action]] before [[DUE:due]].",
    "Open action item: [[OWNER:owner]] to [[ACTION:action]] (due [[DUE:due]]).",
    "[[OWNER:owner]] will [[ACTION:action]].",
    "[[OWNER:owner]] is to [[ACTION:action]] no later than [[DUE:due]].",
]
FORMAL_VISIT_TITLES = {
    "SIV": ["Site Initiation Visit (SIV)", "Site Initiation Visit"],
    "IMV": ["Interim Monitoring Visit (IMV)", "Routine Monitoring Visit", "Interim Monitoring Visit"],
    "REMOTE": ["Remote Monitoring Visit (RMV)", "Remote Interim Monitoring Visit"],
    "COV": ["Close-Out Visit (COV)", "Close-Out Visit"],
    "FOR_CAUSE": ["Directed/For-Cause Monitoring Visit", "For-Cause Monitoring Visit", "For-Cause Visit"],
}
FORMAL_CREDENTIALS = ["", ", RN, CCRA", ", PhD", ", CCRC", ", MS", ", CCRA", ", BSN"]
FORMAL_DATE_FORMATS = ["%B %d, %Y", "%d-%b-%Y", "%d %B %Y", "%m/%d/%Y", "%B %d, %Y"]
FORMAL_ENROLLMENT = [
    "To date, [[SCREENED:ns]] subjects have been screened and [[ENROLLED:ne]] have been randomized.",
    "Subject Status: [[ENROLLED:ne]] subjects are randomized out of [[SCREENED:ns]] screened.",
    "The site has screened [[SCREENED:ns]] subjects and randomized [[ENROLLED:ne]].",
    "Enrollment stands at [[ENROLLED:ne]] randomized subjects ([[SCREENED:ns]] screened).",
]
FORMAL_TOPICS = {
    "DATA_ENTRY_BACKLOG": "Data Management", "QUERY_AGING": "Data Queries", "SDV_BACKLOG": "Source Data Verification",
    "SAE_REPORTING": "Safety", "CONSENT": "Informed Consent", "PROTOCOL_DEVIATION": "Protocol Compliance",
    "IP_ACCOUNTABILITY": "IP Accountability", "TEMP_EXCURSION": "IP Storage", "STAFF_TURNOVER": "Site Staff",
    "PI_OVERSIGHT": "Investigator Oversight", "ENROLLMENT_LAG": "Enrollment", "REG_DOCS": "Regulatory Binder",
}
FORMAL_OWNERS = ["The SC", "The site", "The Study Coordinator", "The PI", "The regulatory coordinator"]
FORMAL_RELATIVE_DUES = ["next Friday", "the next visit", "the end of the month", "the next monitoring visit"]
# completed items written under an "Action Item" label: not open follow-ups
FORMAL_COMPLETED = [
    "The PI signed the updated delegation log during the visit.",
    "The SC filed the renewed laboratory certification.",
    "The site returned the signed site visit log.",
    "The pharmacist completed the drug return form.",
    "The PI signed off on the eligibility checklist for the newest subject.",
]


# ---------------------------------------------------------------------------
# Severity rubric v2 (agreed with a clinical research reviewer)
#   critical (high risk on its own): SAE unreported or reported late; procedures before consent or no signed
#     ICF; ineligible subject dosed; dosing errors; expired IP or IP used after an excursion before assessment;
#     enrolling after IRB approval lapsed; untrained staff running visits; refusal of source access
#   major (1 = medium, 2+ = high): outdated ICF / re-consent overdue, important deviations, kits unaccounted
#     for, unreported excursions, staff not on delegation log, PI not signing labs or casebooks, large data or
#     query backlogs, missing essential documents, enrollment far behind
#   minor (3+ = medium): single out-of-window visit, con-med not entered, a few pages or queries, etc.
# Critical findings stay active even when a CAPA is in place.
# The moves below re-file sentences whose severity changed from the first version.
# ---------------------------------------------------------------------------
RUBRIC_V2_MOVES = [
    # (issue, from, to, substring or "*" for every sentence in the list)
    ("SAE_REPORTING", "major", "critical", "*"),
    ("SAE_REPORTING", "minor", "major", "*"),
    ("DATA_ENTRY_BACKLOG", "critical", "major", "*"),
    ("QUERY_AGING", "critical", "major", "*"),
    ("PI_OVERSIGHT", "critical", "major", "*"),
    ("ENROLLMENT_LAG", "critical", "major", "*"),
    ("IP_ACCOUNTABILITY", "critical", "major", "missing from the pharmacy"),
    ("TEMP_EXCURSION", "critical", "major", "reezer failure"),
    ("PROTOCOL_DEVIATION", "major", "critical", "dosed despite"),
]


def _apply_rubric_v2(bank: dict[str, dict[str, list[str]]]) -> None:
    for code, src, dst, needle in RUBRIC_V2_MOVES:
        moving = [s for s in bank[code][src] if needle == "*" or needle in s]
        bank[code][src] = [s for s in bank[code][src] if s not in moving]
        bank[code][dst] = bank[code][dst] + moving


_apply_rubric_v2(ISSUE_SENTENCES)
_apply_rubric_v2(FORMAL_ISSUE_SENTENCES)
