import random
import unittest

from scope.data import templates as T
from scope.data.generate import generate, sample_note
from scope.data.handwritten import load_handwritten, load_realistic
from scope.record import assemble_actions
from scope.schema import BIO_LABELS, ISSUE_CODES, RISK_LEVELS
from scope.text import bio_to_spans, tokenize


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate(20, seed=1), generate(20, seed=1))

    def test_rows_are_consistent(self):
        for row in generate(200, seed=3) + generate(100, seed=4, unseen=True):
            self.assertIn(row["risk"], RISK_LEVELS)
            self.assertTrue(set(row["issues"]) <= set(ISSUE_CODES))
            self.assertFalse(set(row["issues"]) & set(row["inactive"]))
            self.assertEqual(len(row["tokens"]), len(row["tags"]))
            self.assertTrue(set(row["tags"]) <= set(BIO_LABELS))
            for sp in row["spans"]:
                self.assertEqual(row["text"][sp["char_start"]:sp["char_end"]], sp["text"])
            # gold action items are exactly what the record builder assembles from the gold spans
            spans = bio_to_spans(tokenize(row["text"]), row["tags"], row["text"])
            got = [{k: a[k] for k in ("action", "owner", "due")} for a in assemble_actions(row["text"], spans)]
            self.assertEqual(got, row["actions"])

    def test_email_style_only_in_unseen(self):
        seen = {r["style"] for r in generate(300, seed=5)}
        self.assertNotIn("email", seen)
        rng = random.Random(0)
        self.assertEqual(sample_note(rng, unseen=True, style="email")["style"], "email")

    def test_formal_style(self):
        rows = [sample_note(random.Random(i), style="formal") for i in range(30)]
        self.assertTrue(all(r["style"] == "formal" for r in rows))
        self.assertTrue(any("Visit Type:" in r["text"] for r in rows))
        # formal notes are mostly clean: several topics are mentioned without being active issues
        self.assertGreater(sum(len(r["inactive"]) for r in rows), sum(len(r["issues"]) for r in rows))

    def test_split_variants(self):
        _seen, held = T.split_variants(list(range(8)))
        self.assertEqual(held, [3, 7])
        self.assertEqual(T.split_variants([1, 2]), ([1, 2], []))


class TestHandwritten(unittest.TestCase):
    def test_realistic(self):
        rows = load_realistic()
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r["style"] == "realistic" for r in rows))


    def test_load(self):
        rows = load_handwritten()
        self.assertGreaterEqual(len(rows), 20)
        for r in rows:
            self.assertIn(r["risk"], RISK_LEVELS)
            self.assertTrue(set(r["issues"]) <= set(ISSUE_CODES), r["id"])
            self.assertEqual(len(r["tokens"]), len(r["tags"]))
            labelled = [t for t in r["tags"] if t != "O"]
            self.assertTrue(labelled, r["id"])


if __name__ == "__main__":
    unittest.main()
