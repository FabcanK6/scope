import unittest

from scope.data.generate import generate

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
class TestSearch(unittest.TestCase):
    def test_tfidf_search_and_filter(self):
        from scope.search import NoteIndex, evaluate_retrieval

        rows = generate(80, seed=11)
        index = NoteIndex(rows, backend="tfidf")
        hits = index.search(rows[0]["text"], k=3)
        self.assertEqual(hits[0]["id"], rows[0]["id"])
        hits = index.search("queries", k=5, issue_filter=["QUERY_AGING"])
        self.assertTrue(all("QUERY_AGING" in h["issues"] for h in hits))
        res = evaluate_retrieval(index, generate(20, seed=12), k=3)
        self.assertIn("precision@3", res)


if __name__ == "__main__":
    unittest.main()
