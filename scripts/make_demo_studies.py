"""Build SCOPE's demo studies: five fictional protocols (PDF), a study profile for each, and example notes.

    python scripts/make_demo_studies.py      # writes profiles/demos/<study>/{protocol.pdf, profile.json, notes.json}

Every rule in a demo profile quotes its protocol word for word (with the page), exactly as SCOPE's protocol reader
produces them, so the demos show what a profile built from a protocol looks like without needing an AI call.
All studies, drugs, devices, sponsors, sites and people are invented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scope import profile as P  # noqa: E402
from scope.protocol import _page_of, read_document, rule_line  # noqa: E402

OUT = ROOT / "profiles" / "demos"
FICTION = "FICTIONAL EXAMPLE PROTOCOL - FOR SOFTWARE DEMONSTRATION ONLY. All names and details are invented."

# Each study: pages of (heading, [sentences]); rules point at a sentence by its text (quote = that sentence).
STUDIES = [
 {
  "id": "zlv-301", "label": "Psoriasis · ZLV-301 (Phase III)", "number": "ZLV-301", "version": "Amendment 2",
  "title": "A Phase III, Randomized, Double-blind, Placebo-controlled Study of Zelvatinib in Adults with "
           "Moderate-to-Severe Plaque Psoriasis",
  "phase": "III", "area": "Dermatology",
  "summary": "Oral tablets, adults with psoriasis. SAEs within 2 business days; elective pre-planned "
             "hospitalizations are not SAEs; visit windows ±3 days to Week 12.",
  "pages": [
   ("Protocol ZLV-301, Amendment 2 (Version 3.0)", [FICTION,
     "Primary endpoint: proportion of subjects achieving PASI 75 at Week 12.",
     "The Week 12 PASI assessment is the primary efficacy assessment."]),
   ("6. Study Population", [
     "Subjects who do not meet all eligibility criteria must not be randomized.",
     "Randomization of an ineligible subject is an important protocol deviation."]),
   ("7. Study Drug", [
     "Zelvatinib tablets must be stored at 15 to 25 C.",
     "Any temperature excursion must be reported to the sponsor within 1 business day, and affected kits must be "
     "quarantined and not dispensed until the sponsor approves their use.",
     "Study drug must be interrupted if ALT or AST rises above 3 times the upper limit of normal.",
     "Returned tablets must be counted at every dispensing visit."]),
   ("8. Schedule of Assessments", [
     "Visits from Week 2 to Week 12 must occur within plus or minus 3 days of the scheduled day; visits from Week 16 "
     "onward within plus or minus 7 days.",
     "A missed Week 12 PASI assessment is an important protocol deviation."]),
   ("10. Safety Reporting", [
     "Worsening of psoriasis is recorded as an adverse event only if it is more severe than the subject's usual "
     "fluctuation.",
     "A hospitalization for an elective procedure that was planned before the subject signed the informed consent "
     "form is not a serious adverse event.",
     "All serious adverse events must be reported to the sponsor within 2 business days of the site becoming aware "
     "of the event.",
     "Any pregnancy in a subject must be reported to the sponsor within 24 hours of the site becoming aware of it."]),
  ],
  "rules": [
   ("ELIGIBILITY", "critical", "Randomization of an ineligible subject is an important protocol deviation.", None),
   ("TEMP_EXCURSION", "critical", "Any temperature excursion must be reported to the sponsor within 1 business day, "
    "and affected kits must be quarantined and not dispensed until the sponsor approves their use.", None),
   ("DOSING_ERROR", "critical", "Study drug must be interrupted if ALT or AST rises above 3 times the upper limit of "
    "normal.", None),
   ("DOSING_ERROR", "major", "Returned tablets must be counted at every dispensing visit.", None),
   ("PROTOCOL_DEVIATION", "minor", "Visits from Week 2 to Week 12 must occur within plus or minus 3 days of the "
    "scheduled day; visits from Week 16 onward within plus or minus 7 days.", None),
   ("PROTOCOL_DEVIATION", "major", "A missed Week 12 PASI assessment is an important protocol deviation.", None),
   ("AE_REPORTING", "definition", "Worsening of psoriasis is recorded as an adverse event only if it is more severe "
    "than the subject's usual fluctuation.", None),
   ("SAE_REPORTING", "definition", "A hospitalization for an elective procedure that was planned before the subject "
    "signed the informed consent form is not a serious adverse event.", None),
   ("SAE_REPORTING", "critical", "All serious adverse events must be reported to the sponsor within 2 business days "
    "of the site becoming aware of the event.", (2, "business_days")),
   ("SAFETY_REPORTS", "critical", "Any pregnancy in a subject must be reported to the sponsor within 24 hours of the "
    "site becoming aware of it.", None),
  ],
  "notes": [
   ("SAE timing under this protocol", "low",
    "The SAE was reported 2 business days after awareness (Thursday to Monday), which this protocol allows; the knee "
    "replacement was planned before consent, so it is not an SAE.",
    "IMV, Site 112 (study ZLV-301), 8 October 2026. Subject 112-004 was hospitalized for pneumonia; the site became "
    "aware on Thursday 1 October 2026 and reported the SAE to the sponsor on Monday 5 October 2026. Subject 112-009 "
    "was admitted on 28 September 2026 for a knee replacement that was scheduled before she signed the informed "
    "consent form; no SAE form was filed. Week 12 PASI assessments were completed for all subjects within the visit "
    "window. Drug storage logs stayed within 15-25 C."),
   ("Quarantined kit dispensed", "high",
    "Drug from kits quarantined after an excursion was dispensed before the sponsor approved it: critical.",
    "ZLV-301 IMV - Site 118 - 14 Oct 2026 - monitor: L. Varga\n- Pharmacy fridge room hit 29 C on 22 Sep; kits 3301-"
    "3310 quarantined and excursion reported to sponsor 23 Sep.\n- Kit 3304 from the quarantined batch was dispensed "
    "to subject 118-015 on 30 Sep, before any sponsor decision.\n- Subject has taken 14 days of tablets, no AEs.\n- "
    "Pharmacist to retrieve remaining tablets and file a deviation by 16 Oct 2026."),
   ("Late visit inside the window, uncounted returns", "medium",
    "A Week 16 visit 6 days late is inside this protocol's ±7-day window, so it is not a deviation; returned tablets "
    "were not counted at two dispensing visits (dosing compliance, major).",
    "Interim visit, Site 130, ZLV-301, 21 October 2026. Subject 130-002's Week 16 visit took place 6 days after the "
    "scheduled day because of travel. Returned tablets were not counted at the Week 8 and Week 12 dispensing visits "
    "for subjects 130-002 and 130-006; the coordinator says the bottles were discarded. Consents and PASI "
    "assessments were in order. Coordinator to count returns at every dispensing visit from now on."),
  ],
 },
 {
  "id": "onc-210", "label": "Oncology · ONC-210 (Phase II)", "number": "ONC-210", "version": "Version 2.0",
  "title": "A Phase II, Open-label Study of Rovanimab plus Carboplatin in Previously Treated Non-Small Cell Lung "
           "Cancer",
  "phase": "II", "area": "Oncology",
  "summary": "IV infusion, NSCLC. Progression is not an SAE; scans every 6 weeks ±7 days; dosing held if "
             "neutrophils < 1.0; SAEs within 24 hours.",
  "pages": [
   ("Protocol ONC-210, Version 2.0", [FICTION,
     "Primary endpoint: objective response rate by RECIST 1.1."]),
   ("5. Study Treatment", [
     "Rovanimab vials must be stored at 2 to 8 C and must not be frozen.",
     "Treatment must be held if the absolute neutrophil count is below 1.0 x 10^9/L on the day of dosing.",
     "Dosing a subject whose absolute neutrophil count is below 1.0 x 10^9/L is an important protocol deviation."]),
   ("7. Assessments", [
     "Tumour assessments (CT scans) are performed every 6 weeks plus or minus 7 days.",
     "A tumour assessment outside its window, or missed, is an important protocol deviation.",
     "Pharmacokinetic samples are collected before and at the end of the Cycle 1 Day 1 infusion."]),
   ("9. Adverse Events", [
     "Disease progression is an efficacy outcome and must not be reported as an adverse event or serious adverse "
     "event.",
     "Hospitalization due only to disease progression is not a serious adverse event.",
     "All serious adverse events must be reported to the sponsor within 24 hours of the site becoming aware of the "
     "event.",
     "Infusion-related reactions of grade 3 or higher are adverse events of special interest and must be reported "
     "within 24 hours."]),
   ("11. Informed Consent", [
     "When a new version of the informed consent form is approved, ongoing subjects must be re-consented at their "
     "next visit."]),
  ],
  "rules": [
   ("TEMP_EXCURSION", "major", "Rovanimab vials must be stored at 2 to 8 C and must not be frozen.", None),
   ("DOSING_ERROR", "critical", "Dosing a subject whose absolute neutrophil count is below 1.0 x 10^9/L is an "
    "important protocol deviation.", None),
   ("PROTOCOL_DEVIATION", "major", "A tumour assessment outside its window, or missed, is an important protocol "
    "deviation.", None),
   ("PROTOCOL_DEVIATION", "definition", "Tumour assessments (CT scans) are performed every 6 weeks plus or minus 7 "
    "days.", None),
   ("AE_REPORTING", "definition", "Disease progression is an efficacy outcome and must not be reported as an adverse "
    "event or serious adverse event.", None),
   ("SAE_REPORTING", "definition", "Hospitalization due only to disease progression is not a serious adverse event.",
    None),
   ("SAE_REPORTING", "critical", "All serious adverse events must be reported to the sponsor within 24 hours of the "
    "site becoming aware of the event.", (24, "hours")),
   ("SAFETY_REPORTS", "critical", "Infusion-related reactions of grade 3 or higher are adverse events of special "
    "interest and must be reported within 24 hours.", None),
   ("CONSENT", "major", "When a new version of the informed consent form is approved, ongoing subjects must be "
    "re-consented at their next visit.", None),
  ],
  "notes": [
   ("Hospitalized for progression", "low",
    "A hospitalization due only to disease progression is not an SAE under this protocol, and the scan was inside its "
    "window.",
    "Monitoring visit ONC-210, Site 21, 12 October 2026. Subject 21-007 was admitted on 2 October 2026 with "
    "worsening shortness of breath; the discharge summary attributes it to disease progression only, and the site "
    "recorded it as progression, not as an SAE. Her Week 12 CT scan was done 5 days after the scheduled date. All "
    "subjects are on the current ICF version. Vials stored at 2-8 C throughout."),
   ("Dosed with low neutrophils", "high",
    "Cycle 3 was given with an ANC of 0.8 x 10^9/L, which the protocol says must hold treatment: critical.",
    "ONC-210 / Site 34 / 07-Oct-2026\n- 34-002 received Cycle 3 rovanimab + carboplatin on 29-Sep with ANC 0.8 x "
    "10^9/L (lab drawn same morning). PI says the result was not reviewed before dosing.\n- No infusion reactions.\n"
    "- Scans on schedule.\n- PI to report the dosing deviation to the sponsor and IRB by 9-Oct-2026."),
   ("Late scan", "medium",
    "A tumour assessment 11 days late is outside the ±7-day window: an important deviation (major).",
    "IMV, ONC-210, Site 40, 15 October 2026. Subject 40-005's Week 18 CT scan was performed 11 days after the "
    "scheduled date because the scanner was down. Everything else reviewed (consents, SAEs, drug accountability, "
    "storage logs) was in order. Site to document the late scan as a deviation."),
  ],
 },
 {
  "id": "vax-118", "label": "Vaccine · VAX-118 (Phase III)", "number": "VAX-118", "version": "Version 1.2",
  "title": "A Phase III, Observer-blind Study of an Adjuvanted RSV Vaccine (RSVa-2) in Adults Aged 60 and Over",
  "phase": "III", "area": "Vaccines",
  "summary": "Single injection, adults 60+. Strict cold chain 2-8 C; Day 29 blood sample is key; 7-day e-diary; "
             "SAEs within 24 hours.",
  "pages": [
   ("Protocol VAX-118, Version 1.2", [FICTION,
     "Primary endpoint: RSV neutralizing antibody titres at Day 29."]),
   ("6. Vaccine Handling", [
     "Vaccine must be stored at 2 to 8 C at all times.",
     "Any temperature excursion must be reported to the sponsor within 1 calendar day, and affected doses must be "
     "quarantined and must not be administered until the sponsor approves their use.",
     "Administering a dose exposed to an unapproved temperature excursion is an important protocol deviation."]),
   ("7. Schedule", [
     "The Day 29 visit must occur within plus or minus 3 days.",
     "Blood samples for immunogenicity are taken before vaccination on Day 1 and at Day 29.",
     "A missed Day 29 immunogenicity sample is an important protocol deviation."]),
   ("8. Reactogenicity", [
     "Subjects record solicited reactions in an electronic diary for 7 days after vaccination.",
     "Diary completion on fewer than 5 of the 7 days requires retraining of the subject and is a minor deviation."]),
   ("9. Safety Reporting", [
     "All serious adverse events must be reported to the sponsor within 24 hours of the site becoming aware of the "
     "event.",
     "Guillain-Barre syndrome is an adverse event of special interest and must be reported within 24 hours."]),
  ],
  "rules": [
   ("TEMP_EXCURSION", "critical", "Administering a dose exposed to an unapproved temperature excursion is an "
    "important protocol deviation.", None),
   ("TEMP_EXCURSION", "major", "Any temperature excursion must be reported to the sponsor within 1 calendar day, and "
    "affected doses must be quarantined and must not be administered until the sponsor approves their use.",
    (1, "calendar_days")),
   ("PROTOCOL_DEVIATION", "minor", "The Day 29 visit must occur within plus or minus 3 days.", None),
   ("PROTOCOL_DEVIATION", "major", "A missed Day 29 immunogenicity sample is an important protocol deviation.", None),
   ("PROTOCOL_DEVIATION", "minor", "Diary completion on fewer than 5 of the 7 days requires retraining of the subject "
    "and is a minor deviation.", None),
   ("SAE_REPORTING", "critical", "All serious adverse events must be reported to the sponsor within 24 hours of the "
    "site becoming aware of the event.", (24, "hours")),
   ("SAFETY_REPORTS", "critical", "Guillain-Barre syndrome is an adverse event of special interest and must be "
    "reported within 24 hours.", None),
  ],
  "notes": [
   ("Diary gaps, visit inside window", "low",
    "One subject completed the diary on 4 of 7 days (minor); a Day 29 visit on Day 31 is inside the ±3-day window.",
    "Hi team, notes from the VAX-118 visit at Site 7 on 9 October 2026. Subject 07-044 completed her e-diary on "
    "only 4 of the 7 days; she was retrained during the visit. Subject 07-051 attended his Day 29 visit on Day 31. "
    "All Day 1 and Day 29 blood samples were collected. Fridge logs 2-8 C, no excursions. No SAEs. Thanks, Ana"),
   ("Doses given after a fridge excursion", "high",
    "Doses from a fridge that reached 11 C were given before any sponsor decision: critical.",
    "VAX-118 IMV, Site 15, 13 Oct 2026. Clinic fridge reached 11 C for about 5 hours on 6 Oct (power cut). The "
    "excursion was not reported and the doses were not quarantined. Three subjects (15-020, 15-021, 15-022) were "
    "vaccinated from that fridge on 7 Oct. Site to report the excursion and the three administrations to the "
    "sponsor today."),
   ("Missed Day 29 sample", "medium",
    "The Day 29 immunogenicity sample is the primary endpoint; missing it is an important deviation (major).",
    "Visit report VAX-118, Site 22, 16 October 2026. Subject 22-013's Day 29 blood sample was not collected because "
    "the lab kit had expired; the visit itself was on time. Diaries complete for all subjects. No excursions, no "
    "SAEs. Lab coordinator to check kit expiry dates weekly."),
  ],
 },
 {
  "id": "crd-07", "label": "Medical device · CRD-07 (pivotal)", "number": "CRD-07", "version": "Revision C",
  "title": "A Prospective, Multicentre Pivotal Study of the HeartLoop Implantable Cardiac Monitor",
  "phase": "Pivotal (device)", "area": "Cardiology",
  "summary": "Implanted device. SAEs and device deficiencies within 3 calendar days; only trained, listed "
             "implanters; serial numbers tracked.",
  "pages": [
   ("Protocol CRD-07, Revision C", [FICTION,
     "Primary endpoint: detection of atrial fibrillation episodes over 12 months."]),
   ("5. Investigators and Training", [
     "Only physicians who have completed HeartLoop implant training and are listed on the delegation log may "
     "perform implants.",
     "An implant performed by a physician who is not trained and listed is an important protocol deviation."]),
   ("6. Device Accountability", [
     "The serial number of every device received, implanted, explanted or returned must be recorded in the device "
     "accountability log."]),
   ("8. Follow-up", [
     "Follow-up visits take place at 3, 6 and 12 months after implant, each within plus or minus 14 days."]),
   ("10. Safety Reporting", [
     "Serious adverse events and device deficiencies that could have led to a serious adverse event must be "
     "reported to the sponsor within 3 calendar days of the site becoming aware of them."]),
  ],
  "rules": [
   ("STAFF_TURNOVER", "critical", "An implant performed by a physician who is not trained and listed is an important "
    "protocol deviation.", None),
   ("IP_ACCOUNTABILITY", "major", "The serial number of every device received, implanted, explanted or returned must "
    "be recorded in the device accountability log.", None),
   ("PROTOCOL_DEVIATION", "minor", "Follow-up visits take place at 3, 6 and 12 months after implant, each within plus "
    "or minus 14 days.", None),
   ("SAE_REPORTING", "critical", "Serious adverse events and device deficiencies that could have led to a serious "
    "adverse event must be reported to the sponsor within 3 calendar days of the site becoming aware of them.",
    (3, "calendar_days")),
  ],
  "notes": [
   ("SAE reported on day 3", "low",
    "Reported 3 calendar days after awareness, which this device protocol allows (the default 24-hour rule would "
    "call it late).",
    "CRD-07 IMV - Site 3 - 12 October 2026. Subject 03-017 had a pocket infection requiring hospitalization; the "
    "site became aware on 5 October 2026 and reported the SAE to the sponsor on 8 October 2026. Device "
    "accountability log complete with all serial numbers. 6-month follow-ups all within window."),
   ("Implant by an untrained physician", "high",
    "A device was implanted by a physician who had not completed implant training and was not on the log: critical.",
    "CRD-07, Site 9, 14 Oct 2026, monitoring visit. Subject 09-004's HeartLoop was implanted on 1 October by Dr. "
    "Okafor, a new cardiology fellow who has not completed the implant training and is not on the delegation log. "
    "The implant was uneventful. PI to report the deviation and schedule Dr. Okafor's training by 21 Oct 2026."),
   ("Serial number missing", "medium",
    "One implanted device's serial number is missing from the accountability log: major under this protocol.",
    "Remote review CRD-07, Site 12, 16 October 2026. The device accountability log has no serial number recorded "
    "for the device implanted in subject 12-008 on 18 September 2026; the implant report shows it. All follow-ups on "
    "time, no SAEs. Coordinator to update the log."),
  ],
 },
 {
  "id": "ped-44", "label": "Pediatrics · PED-44 (Phase II)", "number": "PED-44", "version": "Amendment 1",
  "title": "A Phase II Study of Kalvexin Oral Solution as Add-on Therapy in Children Aged 4 to 17 with Focal Epilepsy",
  "phase": "II", "area": "Pediatric neurology",
  "summary": "Children 4-17. Parent consent plus child assent from age 7; dose by weight each visit; status "
             "epilepticus is always an SAE.",
  "pages": [
   ("Protocol PED-44, Amendment 1", [FICTION,
     "Primary endpoint: change in monthly focal seizure frequency."]),
   ("4. Consent and Assent", [
     "Written informed consent from a parent or legal guardian and written assent from children aged 7 or older "
     "must be obtained before any study procedure.",
     "Study procedures performed before written assent is obtained from a child aged 7 or older are an important "
     "protocol deviation.",
     "Subjects who turn 18 during the study must give their own written consent at their next visit."]),
   ("6. Dosing", [
     "The dose is calculated from body weight at every visit.",
     "A dose differing by more than 10% from the weight-based dose for more than 7 days is an important protocol "
     "deviation."]),
   ("9. Safety Reporting", [
     "Status epilepticus is always a serious adverse event, whether or not the child is hospitalized.",
     "All serious adverse events must be reported to the sponsor within 24 hours of the site becoming aware of the "
     "event."]),
  ],
  "rules": [
   ("CONSENT", "critical", "Study procedures performed before written assent is obtained from a child aged 7 or "
    "older are an important protocol deviation.", None),
   ("CONSENT", "major", "Subjects who turn 18 during the study must give their own written consent at their next "
    "visit.", None),
   ("DOSING_ERROR", "major", "A dose differing by more than 10% from the weight-based dose for more than 7 days is an "
    "important protocol deviation.", None),
   ("SAE_REPORTING", "definition", "Status epilepticus is always a serious adverse event, whether or not the child is "
    "hospitalized.", None),
   ("SAE_REPORTING", "critical", "All serious adverse events must be reported to the sponsor within 24 hours of the "
    "site becoming aware of the event.", (24, "hours")),
  ],
  "notes": [
   ("No assent from an 8-year-old", "high",
    "Screening procedures were done on an 8-year-old with parental consent but no written assent: critical.",
    "PED-44 visit, Site 5, 13 October 2026. Subject 05-011 (age 8) had screening bloods and an EEG on 2 October with "
    "the mother's signed consent, but no written assent from the child was obtained. Dosing has not started. "
    "Seizure diaries complete. Site to obtain assent and report the deviation by 15 October 2026."),
   ("Dose not adjusted for weight", "medium",
    "The dose stayed 14% below the weight-based dose for 4 weeks after weight gain: an important deviation (major).",
    "IMV PED-44, Site 8, 15 Oct 2026. Subject 08-006 gained 4 kg by Week 8 but the dose was not recalculated; for "
    "the following 4 weeks the dose was 14% below the weight-based dose. No seizure worsening reported. Consent and "
    "assent documents in order. Pharmacist to recalculate doses at every visit."),
   ("Status epilepticus at home", "high",
    "Status epilepticus is always an SAE under this protocol, even without hospitalization; it was not reported: "
    "critical.",
    "PED-44 IMV - Site 12 - 16 October 2026. Seizure diary for subject 12-003 shows an episode of status epilepticus "
    "on 3 October 2026, treated at home with rescue medication by the parents; the child was not hospitalized. The "
    "site did not report it as an SAE. Other subjects: diaries complete, doses correct."),
  ],
 },
]


def build_pdf(study: dict, path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    ss = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter, title=f"Protocol {study['number']} (fictional)")
    story = [Paragraph(study["title"], ss["Title"])]
    for i, (head, sentences) in enumerate(study["pages"]):
        story.append(Paragraph(head, ss["Heading2"]))
        for t in sentences:
            story += [Paragraph(t, ss["BodyText"]), Spacer(1, 6)]
        if i < len(study["pages"]) - 1:
            story.append(PageBreak())
    doc.build(story)


def build_profile(study: dict, pages: list[str]) -> dict:
    p = P.default_profile()
    ref = f"{study['number']} {study['version']}"
    rules = []
    for topic, severity, quote, deadline in study["rules"]:
        page = _page_of(quote, pages)
        if page is None:
            raise SystemExit(f"{study['id']}: quote not found in the PDF: {quote[:60]}")
        rule = {"topic": topic, "rule": quote, "severity": severity, "quote": quote, "page": page,
                "deadline": {"amount": deadline[0], "unit": deadline[1]} if deadline else None}
        rules.append(rule)
        if deadline:
            p["deadlines"] = [d for d in p["deadlines"] if d["topic"] != topic] + [
                {"topic": topic, "amount": deadline[0], "unit": deadline[1], "severity": severity, "what": quote,
                 "source": f"protocol {ref}, p. {page}"}]
    p["study_rules"] = [rule_line(r, ref) for r in rules]
    p.update(name=f"Demo · {study['label']}", version="1", description=study["summary"],
             protocol={"file": "protocol.pdf", "reference": ref, "title": study["title"], "phase": study["phase"],
                       "therapeutic_area": study["area"], "rules": rules},
             demo={"id": study["id"], "label": study["label"]})
    P.log_change(p, f"Built from fictional protocol {ref}: {len(rules)} rules", who="SCOPE demo")
    return P.validate(p)


def main() -> None:
    index = []
    for study in STUDIES:
        folder = OUT / study["id"]
        folder.mkdir(parents=True, exist_ok=True)
        pdf = folder / "protocol.pdf"
        build_pdf(study, pdf)
        pages = read_document("protocol.pdf", pdf.read_bytes())
        prof = build_profile(study, pages)
        (folder / "profile.json").write_text(P.dumps(prof) + "\n")
        notes = [{"id": f"{study['id']}-{i + 1}", "title": t, "expected_risk": r, "why": why, "text": text}
                 for i, (t, r, why, text) in enumerate(study["notes"])]
        (folder / "notes.json").write_text(json.dumps(notes, indent=2) + "\n")
        index.append({"id": study["id"], "label": study["label"], "summary": study["summary"],
                      "area": study["area"], "notes": len(notes), "rules": len(prof["study_rules"])})
        print(f"{study['id']}: {len(pages)} pages, {len(prof['study_rules'])} rules, {len(notes)} notes")
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
