"""Demo studies: every rule quotes its protocol, profiles load, notes cover every risk level."""
import unittest

from scope import demos as DM
from scope import profile as P
from scope.engine import build_system
from scope.protocol import read_document, verify_quote


class TestDemos(unittest.TestCase):
    def test_index(self):
        ids = [s["id"] for s in DM.index()]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(s["area"] for s in DM.index())), 5)  # five different therapeutic areas

    def test_profiles_quote_their_protocols(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf not installed")
        for s in DM.index():
            prof = DM.profile(s["id"])
            pages = read_document("protocol.pdf", DM.protocol_path(s["id"]).read_bytes())
            self.assertTrue(prof["protocol"]["rules"], s["id"])
            for r in prof["protocol"]["rules"]:
                self.assertTrue(verify_quote(r["quote"], pages[r["page"] - 1]), (s["id"], r["quote"][:50]))
            self.assertEqual(len(prof["study_rules"]), len(prof["protocol"]["rules"]))
            self.assertIn("override the rubric", build_system(prof))
            self.assertEqual(P.loads(P.dumps(prof))["name"], prof["name"])

    def test_deadlines_differ_by_study(self):
        sae = {s["id"]: P.deadline_map(DM.profile(s["id"]))["SAE_REPORTING"] for s in DM.index()}
        self.assertEqual((sae["zlv-301"]["amount"], sae["zlv-301"]["unit"]), (2, "business_days"))
        self.assertEqual((sae["crd-07"]["amount"], sae["crd-07"]["unit"]), (3, "calendar_days"))
        self.assertEqual((sae["onc-210"]["amount"], sae["onc-210"]["unit"]), (24, "hours"))

    def test_notes(self):
        rows = DM.labelled_rows()
        self.assertEqual(len(rows), 15)
        self.assertEqual({r["risk"] for r in rows}, {"low", "medium", "high"})
        for r in rows:
            self.assertTrue(r["text"] and r["why"] and r["study"])



class TestLiveAnswers(unittest.TestCase):
    """Answers the live model gave on demo notes, replayed through SCOPE's checks."""

    def test_ellipsis_quote_keeps_the_unreported_sae(self):
        from scope import demos as DM
        from scope.engine import LLMParser
        from tests.test_llm import FakeClient, fake_response

        note = next(n for n in DM.notes("ped-44") if n["id"] == "ped-44-3")
        answer = {"findings": [{
            "issue": "SAE_REPORTING", "status": "active", "severity": "critical",
            "evidence": "Seizure diary for subject 12-003 shows an episode of status epilepticus on 3 October 2026... "
                        "The site did not report it as an SAE.",
            "explanation": "Status epilepticus is an SAE under the study rules.", "repeat": False,
            "subjects_affected": 1, "site_wide": False, "escalation_evidence": "", "clock_start": "3 October 2026",
            "reported_on": None}],
            "actions": [], "summary": "Unreported SAE.",
            "visit": {"enrolled": None, "monitor": None, "pi": None, "screened": None, "site": "Site 12",
                      "visit_date": "16 October 2026", "visit_type": "IMV"}}
        rec = LLMParser(FakeClient([fake_response(answer)]), profile=DM.profile("ped-44")).analyze(note["text"])
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertTrue(rec["findings"][0]["verified"])
        self.assertGreaterEqual(rec["findings"][0]["char_start"], 0)  # highlighted from the first part to the last
        self.assertGreater(rec["findings"][0]["char_end"], rec["findings"][0]["char_start"])


    def test_protocol_rule_sets_the_severity(self):
        from scope import demos as DM
        from scope.engine import LLMParser, build_system
        from tests.test_llm import FakeClient, fake_response

        prof = DM.profile("onc-210")
        n = next(i for i, r in enumerate(prof["study_rules"], 1) if r.startswith("A tumour assessment outside"))
        self.assertIn(f"R{n}: A tumour assessment outside its window", build_system(prof))
        note = next(x for x in DM.notes("onc-210") if x["id"] == "onc-210-3")
        finding = {"issue": "PROTOCOL_DEVIATION", "status": "active", "severity": "minor",
                   "evidence": "Subject 40-005's Week 18 CT scan was performed 11 days after the scheduled date "
                               "because the scanner was down.",
                   "explanation": "Out-of-window scan.", "repeat": False, "subjects_affected": 1, "site_wide": False,
                   "escalation_evidence": "", "clock_start": None, "reported_on": None}
        answer = {"findings": [{**finding, "study_rule": n}], "actions": [], "summary": "Late scan.",
                  "visit": {"enrolled": None, "monitor": None, "pi": None, "screened": None, "site": "Site 40",
                            "visit_date": "15 October 2026", "visit_type": "IMV"}}
        rec = LLMParser(FakeClient([fake_response(answer)]), profile=prof).analyze(note["text"])
        self.assertEqual(rec["risk"]["level"], "medium")  # live: the model said minor, the protocol says major
        self.assertIn("study rule R", rec["findings"][0]["rule_applied"])
        self.assertIn("p. 3", rec["findings"][0]["rule_applied"])
        # a rule for another topic, or no rule, leaves the model's severity alone
        other = next(i for i, r in enumerate(prof["study_rules"], 1) if "[DOSING_ERROR" in r)
        for cited in (other, None, 99, "R" + str(n)):
            ans = {**answer, "findings": [{**finding, "study_rule": cited}]}
            rec = LLMParser(FakeClient([fake_response(ans)]), profile=prof).analyze(note["text"])
            self.assertEqual(rec["risk"]["level"], "medium" if cited == "R" + str(n) else "low", cited)


if __name__ == "__main__":
    unittest.main()
