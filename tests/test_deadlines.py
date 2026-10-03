"""Reporting deadlines are counted by SCOPE from the note's dates, not by the AI model."""
import datetime as dt
import unittest

from scope import deadlines as DL
from scope import profile as P
from scope import protocol as PR
from scope.engine import LLMParser
from tests.test_llm import FakeClient, fake_response

NOTE = ("IMV, Site 112 (study ZLV-301), 8 October 2026. Subject 112-004 was hospitalized for pneumonia; the site "
        "became aware on Thursday 1 October 2026 and reported the SAE to the sponsor on Monday 5 October 2026.")

# what the live model (gemini flash-lite) actually answered: "late" under a 2-business-day rule, which is wrong
LIVE_ANSWER = {
    "visit": {"visit_type": "IMV", "visit_date": "8 October 2026", "site": "Site 112", "monitor": None, "pi": None,
              "screened": None, "enrolled": None},
    "actions": [], "summary": "Late SAE.",
    "findings": [{"issue": "SAE_REPORTING", "status": "active", "severity": "critical",
                  "evidence": "the site became aware on Thursday 1 October 2026 and reported the SAE to the sponsor "
                              "on Monday 5 October 2026",
                  "explanation": "reported outside the 2-business-day window", "repeat": False,
                  "subjects_affected": 1, "site_wide": False, "escalation_evidence": "",
                  "clock_start": "Thursday 1 October 2026", "reported_on": "Monday 5 October 2026"}],
}


def zlv_profile():
    p = P.default_profile()
    p["deadlines"] = [{"topic": "SAE_REPORTING", "amount": 2, "unit": "business_days", "severity": "critical",
                       "what": "SAEs reported to the sponsor", "source": "protocol ZLV-301, p. 5"}]
    return P.validate(p)


class TestDates(unittest.TestCase):
    def test_parse(self):
        oct1 = dt.date(2026, 10, 1)
        for text in ("Thursday 1 October 2026", "Oct 1st, 2026", "01-Oct-2026", "10/01/2026", "2026-10-01",
                     "on 1 October 2026", "October 1, 2026"):
            self.assertEqual(DL.parse_date(text), oct1, text)
        self.assertEqual(DL.parse_date("1 Oct", default_year=2026), oct1)
        self.assertIsNone(DL.parse_date("1 Oct"))
        self.assertIsNone(DL.parse_date("last week"))

    def test_counting(self):
        thu, mon = dt.date(2026, 10, 1), dt.date(2026, 10, 5)
        self.assertEqual(DL.business_days_between(thu, mon), 2)
        self.assertEqual(DL.check(thu, mon, 2, "business_days")["verdict"], "on time")
        self.assertEqual(DL.check(thu, mon, 3, "calendar_days")["verdict"], "late")  # 4 calendar days
        self.assertEqual(DL.check(thu, thu, 24, "hours")["verdict"], "on time")
        self.assertEqual(DL.check(thu, dt.date(2026, 10, 2), 24, "hours")["verdict"], "unclear")
        self.assertEqual(DL.check(thu, dt.date(2026, 10, 3), 24, "hours")["verdict"], "late")
        self.assertEqual(DL.check(mon, thu, 2, "business_days")["verdict"], "unclear")


class TestEngineDeadlines(unittest.TestCase):
    def test_protocol_deadline_overrides_the_model(self):
        rec = LLMParser(FakeClient([fake_response(LIVE_ANSWER)]), profile=zlv_profile()).analyze(NOTE)
        self.assertEqual(rec["risk"]["level"], "low")  # on time under this protocol
        sae = rec["findings"][0]
        self.assertEqual(sae["status"], "no_issue")
        self.assertEqual(sae["deadline_check"]["elapsed"], 2)
        self.assertIn("The AI model said late", rec["checks"][0])
        self.assertIn("2 business days", rec["checks"][0])

    def test_same_note_under_the_default_24_hours(self):
        ans = {**LIVE_ANSWER, "findings": [{**LIVE_ANSWER["findings"][0], "status": "no_issue",
                                            "severity": "minor"}]}  # model wrongly says on time
        rec = LLMParser(FakeClient([fake_response(ans)])).analyze(NOTE)
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertEqual(rec["findings"][0]["final_severity"], "critical")
        self.assertIn("this study allows 24 hours, so it was late", rec["checks"][0])

    def test_dates_not_in_the_note_are_ignored(self):
        ans = {**LIVE_ANSWER, "findings": [{**LIVE_ANSWER["findings"][0], "reported_on": "Friday 2 October 2026"}]}
        rec = LLMParser(FakeClient([fake_response(ans)]), profile=zlv_profile()).analyze(NOTE)
        self.assertEqual(rec["findings"][0]["status"], "active")  # no check: the model's date is not in the note
        self.assertEqual(rec["checks"], [])

    def test_protocol_rules_set_the_deadline(self):
        draft = {"study": {"protocol_number": "ZLV-301", "version": None},
                 "rules": [{"topic": "SAE_REPORTING", "rule": "SAEs within 2 business days.", "severity": "critical",
                            "quote": "q", "page": 5, "deadline": {"amount": 2, "unit": "business_days"}}]}
        prof = PR.apply_rules(P.default_profile(), draft, [0], "zlv.pdf")
        self.assertEqual([(d["topic"], d["amount"], d["unit"]) for d in prof["deadlines"]],
                         [("SAE_REPORTING", 2, "business_days")])
        self.assertIn("within 2 business days", P.rubric_text(prof))


if __name__ == "__main__":
    unittest.main()
