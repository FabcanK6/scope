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


if __name__ == "__main__":
    unittest.main()
