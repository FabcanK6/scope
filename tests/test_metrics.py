import unittest

from scope.data.generate import generate
from scope.metrics import action_prf, evaluate_predictions, expected_calibration_error, headline


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        rows = generate(50, seed=7)
        preds = [{"risk": r["risk"], "issues": r["issues"], "tags": r["tags"], "actions": r["actions"],
                  "risk_probs": {"low": 0.0, "medium": 0.0, "high": 0.0, r["risk"]: 1.0}} for r in rows]
        rep = evaluate_predictions(rows, preds)
        self.assertEqual(rep["risk"]["accuracy"], 1.0)
        self.assertEqual(rep["issues"]["micro"]["f1"], 1.0)
        self.assertEqual(rep["spans"]["micro"]["f1"], 1.0)
        self.assertEqual(rep["note_exact_match"], 1.0)
        self.assertAlmostEqual(rep["risk"]["ece"], 0.0)
        self.assertIn("span_f1", headline(rep))

    def test_action_prf(self):
        g = [[{"action": "a", "owner": "CRC", "due": "x"}]]
        self.assertEqual(action_prf(g, g)["f1"], 1.0)
        self.assertEqual(action_prf(g, [[{"action": "a", "owner": "Site", "due": "x"}]])["f1"], 0.0)

    def test_ece(self):
        self.assertAlmostEqual(expected_calibration_error([0.9, 0.9], [True, False]), 0.4)


if __name__ == "__main__":
    unittest.main()
