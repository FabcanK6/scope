import unittest

from scope.predict import RuleBasedParser
from scope.record import audit_summary, normalize_date, normalize_int, normalize_visit_type, to_row

NOTE = """IMV - Site 104 - 12-Mar-2026
CRA: J. Okafor | PI: Dr. Patel
14 screened, 9 randomized
- 23 queries open > 60 days
- No new protocol deviations noted.
- subject 104-007 underwent study procedures before signing the informed consent form.
CRC to close open queries by 26-Mar-2026.
Action: Site to document the consent deviation (due next visit)."""


class TestNormalizers(unittest.TestCase):
    def test_dates(self):
        for s in ["12-Mar-2026", "12MAR2026", "03/12/2026", "March 12, 2026", "2026-03-12", "12 Mar 2026",
                  "3/12/26", "12.03.2026", "Mar 12, 2026"]:
            self.assertEqual(normalize_date(s), "2026-03-12", s)
        self.assertIsNone(normalize_date("next visit"))

    def test_counts_and_types(self):
        self.assertEqual(normalize_int("nine"), 9)
        self.assertEqual(normalize_int("14"), 14)
        self.assertEqual(normalize_visit_type("site initiation visit"), "SIV")
        self.assertEqual(normalize_visit_type("RMV"), "REMOTE")
        self.assertEqual(normalize_visit_type("Close Out Visit"), "COV")
        self.assertEqual(normalize_visit_type("for-cause visit"), "FOR_CAUSE")
        self.assertEqual(normalize_visit_type("routine monitoring visit"), "IMV")


class TestRulesAndRecord(unittest.TestCase):
    def test_rule_parser_record(self):
        rec = RuleBasedParser().analyze(NOTE)
        v = rec["visit"]
        self.assertEqual(v["site"]["id"], "104")
        self.assertEqual(v["visit_date"]["iso"], "2026-03-12")
        self.assertEqual(v["visit_type"]["code"], "IMV")
        self.assertEqual((v["screened"], v["enrolled"]), (14, 9))
        codes = {i["code"] for i in rec["issues"]}
        self.assertIn("QUERY_AGING", codes)
        self.assertIn("CONSENT", codes)
        self.assertNotIn("PROTOCOL_DEVIATION", codes)  # negated mention
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertEqual(rec["actions"][0], {"action": "close open queries", "owner": "CRC", "due": "26-Mar-2026",
                                             "due_date": "2026-03-26"})
        self.assertIn("Site 104", audit_summary(rec))
        self.assertEqual(to_row(rec)["site"], "104")

    def test_warnings(self):
        rec = RuleBasedParser().analyze("Queries are piling up. Consent missing for one subject.")
        self.assertTrue(any("site" in w.lower() for w in rec["warnings"]))


if __name__ == "__main__":
    unittest.main()
