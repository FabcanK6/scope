"""Visit report drafts and the site history / open actions tracker (fake AI model, no network)."""
import unittest

from scope import reports as RP
from scope import tracker as TR
from scope.engine import LLMParser
from tests.test_llm import FakeClient, fake_response

NOTE_1 = ("IMV Site 12, 3 September 2026. 4 queries open on Week 4 labs. Consent forms all current. "
          "PI to sign the Week 4 lab reports by 10 September 2026.")
NOTE_2 = ("IMV Site 12, 1 October 2026. 6 queries still open on Week 4 labs. No SAEs. "
          "Coordinator to enter Week 8 vitals by 8 October 2026.")


def reading(note, evidence, action, due):
    answer = {"visit": {"visit_type": "IMV", "visit_date": note.split(", ")[1].split(".")[0], "site": "Site 12"},
              "summary": "Queries open.",
              "actions": [{"owner": action.split(" to ")[0], "action": action.split(" to ", 1)[1], "due": due}],
              "findings": [{"issue": "QUERY_AGING", "status": "active", "severity": "minor", "evidence": evidence,
                            "explanation": "open queries", "repeat": False}]}
    return LLMParser(FakeClient([fake_response(answer)])).analyze(note)


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.r1 = reading(NOTE_1, "4 queries open on Week 4 labs", "PI to sign the Week 4 lab reports",
                          "10 September 2026")
        self.r2 = reading(NOTE_2, "6 queries still open on Week 4 labs", "Coordinator to enter Week 8 vitals",
                          "8 October 2026")

    def test_open_actions_carry_to_the_next_visit(self):
        t = {}
        site = TR.site_key(self.r1)
        self.assertEqual(site, "12")
        self.assertTrue(TR.add_visit(t, "Study A", site, TR.visit_entry(self.r1, NOTE_1)))
        e2 = TR.visit_entry(self.r2, NOTE_2)
        TR.add_visit(t, "Study A", site, e2)
        still = TR.open_actions(t, "Study A", site, e2["key"])
        self.assertEqual([a["action"] for a in still], ["sign the Week 4 lab reports"])
        self.assertEqual(still[0]["visit_date"], "2026-09-03")
        self.assertTrue(TR.set_done(t, "Study A", site, still[0]["id"]))
        self.assertEqual(TR.open_actions(t, "Study A", site, e2["key"]), [])
        # reading the same note again keeps the tick and does not add a visit
        self.assertFalse(TR.add_visit(t, "Study A", site, TR.visit_entry(self.r1, NOTE_1)))
        self.assertEqual(len(t["Study A"][site]), 2)
        self.assertEqual(TR.open_actions(t, "Study A", site, e2["key"]), [])
        # other studies are separate
        self.assertEqual(TR.open_actions(t, "Study B", site), [])
        rows = TR.site_summary(t, "Study A")
        self.assertEqual((rows[0]["visits"], rows[0]["last_visit"], rows[0]["open_actions"]), (2, "2026-10-01", 1))

    def test_same_problem_again_is_pointed_out(self):
        t = {}
        TR.add_visit(t, "Study A", "12", TR.visit_entry(self.r1, NOTE_1))
        hints = TR.repeat_hints(t, "Study A", "12", self.r2, NOTE_2)
        self.assertEqual([(h["display"], h["last_date"]) for h in hints], [("Open or aging queries", "2026-09-03")])
        self.assertEqual(TR.repeat_hints({}, "Study A", "12", self.r2, NOTE_2), [])  # first visit: nothing earlier

    def test_visits_sorted_by_date_and_capped(self):
        t = {}
        TR.add_visit(t, "S", "12", TR.visit_entry(self.r2, NOTE_2))
        TR.add_visit(t, "S", "12", TR.visit_entry(self.r1, NOTE_1))
        self.assertEqual([v["date"] for v in t["S"]["12"]], ["2026-09-03", "2026-10-01"])
        for i in range(TR.MAX_VISITS + 5):
            entry = TR.visit_entry(self.r1, NOTE_1 + str(i))
            TR.add_visit(t, "S", "99", {**entry, "date": f"2026-01-{i % 28 + 1:02d}"})
        self.assertEqual(len(t["S"]["99"]), TR.MAX_VISITS)


class TestReport(unittest.TestCase):
    def test_report_uses_only_the_reading_and_flags_gaps(self):
        rec = reading(NOTE_1, "4 queries open on Week 4 labs", "PI to sign the Week 4 lab reports",
                      "10 September 2026")
        facts = RP.report_facts(rec)
        self.assertEqual(facts["active_findings"][0]["evidence"], "4 queries open on Week 4 labs")
        self.assertEqual(facts["action_items"][0]["due"], "2026-09-10")
        draft = ("```markdown\n# Monitoring Visit Report\n| Site | Site 12 |\n## Summary\nQueries open.\n"
                 "## Findings requiring action\n1. **Open or aging queries (minor)**\n## Action items\n"
                 "| Action | Owner | Due |\n## Next visit\n[Next visit date and focus]\n```")
        client = FakeClient([{"candidates": [{"content": {"parts": [{"text": draft}]}}]}])
        report = RP.draft_report(client, rec, NOTE_1)
        self.assertTrue(report.startswith("# Monitoring Visit Report"))  # code fence removed
        self.assertEqual(RP.missing_sections(report), ["Resolved during the visit", "Areas reviewed with no issues"])
        self.assertEqual(RP.placeholders(report), ["Next visit date and focus"])
        self.assertIn("4 queries open on Week 4 labs", client.bodies[-1]["contents"][0]["parts"][0]["text"])


if __name__ == "__main__":
    unittest.main()
