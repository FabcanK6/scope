import unittest

from scope.text import Token, bio_to_spans, char_spans_to_bio, sentences, spans_to_bio, tokenize


class TestText(unittest.TestCase):
    def test_tokenize_offsets(self):
        text = "IMV Site 104 on 12-Mar-2026.\nCRC to fix 3/12/26 queries"
        toks = tokenize(text)
        self.assertIn("12-Mar-2026", [t.text for t in toks])
        self.assertIn("3/12/26", [t.text for t in toks])
        for t in toks:
            self.assertEqual(text[t.start:t.end], t.text)

    def test_char_spans_roundtrip(self):
        text = "Visit at Site 104 by Dr. Patel."
        toks = tokenize(text)
        tags = char_spans_to_bio(toks, [("SITE", 9, 17), ("PI", 21, 30)])
        spans = bio_to_spans(toks, tags, text)
        self.assertEqual([(s.label, s.text) for s in spans], [("SITE", "Site 104"), ("PI", "Dr. Patel")])
        self.assertEqual(spans[1].char_start, 21)

    def test_bio_helpers(self):
        tags = spans_to_bio(5, [("SITE", 1, 3)])
        self.assertEqual(tags, ["O", "B-SITE", "I-SITE", "O", "O"])
        self.assertEqual(bio_to_spans(list("abcde"), tags)[0].as_tuple(), ("SITE", 1, 3))
        # a stray I- starts a new span
        self.assertEqual(len(bio_to_spans(list("ab"), ["I-PI", "O"])), 1)

    def test_sentences_keep_titles(self):
        text = "Met Dr. Patel today. No issues.\n- queries open"
        parts = [text[s:e] for s, e in sentences(text)]
        self.assertEqual(parts, ["Met Dr. Patel today.", "No issues.", "- queries open"])

    def test_token_dataclass(self):
        self.assertEqual(Token("a", 0, 1).end, 1)


if __name__ == "__main__":
    unittest.main()
