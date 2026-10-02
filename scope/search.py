"""Similar-visit search over past notes: sentence embeddings + a vector index.

    from scope.search import NoteIndex
    index = NoteIndex.from_jsonl("data/corpus.jsonl")              # embeddings (all-MiniLM-L6-v2) + FAISS
    hits = index.search("PI not signing labs, queries piling up", k=5)

Backends
--------
``embeddings``  sentence-transformers model, cosine similarity, FAISS ``IndexFlatIP``
                (falls back to NumPy when FAISS is not installed)
``tfidf``       scikit-learn TF-IDF keyword baseline, same interface

``python -m scope.search --eval`` compares the two: a hit is relevant when it
shares an active issue type with the query note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scope.data.generate import read_jsonl

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class NoteIndex:
    def __init__(self, rows: list[dict], backend: str = "embeddings", model_name: str = DEFAULT_EMBEDDER):
        self.rows = rows
        self.backend = backend
        texts = [r["text"] for r in rows]
        if backend == "embeddings":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name)
            vecs = self._embed(texts)
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.encoder = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            vecs = self.encoder.fit_transform(texts).toarray().astype("float32")
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.vectors = vecs
        self.faiss_index = None
        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])
            self.faiss_index.add(vecs)
        except ImportError:
            pass

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> NoteIndex:
        return cls(read_jsonl(path), **kwargs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.backend == "tfidf":
            v = self.encoder.transform(texts).toarray().astype("float32")
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return np.asarray(self.encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                                              show_progress_bar=False), dtype="float32")

    def search(self, query: str, k: int = 5, issue_filter: list[str] | None = None,
               exclude_id: str | None = None) -> list[dict]:
        """Top-k most similar notes. ``issue_filter`` keeps only notes sharing at least one of those issues."""
        q = self._embed([query])
        n = len(self.rows)
        if self.faiss_index is not None:
            scores, idx = self.faiss_index.search(q, n)
            ranked = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = (self.vectors @ q[0]).tolist()
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])
        hits = []
        for i, score in ranked:
            r = self.rows[i]
            if exclude_id and r.get("id") == exclude_id:
                continue
            if issue_filter and not set(issue_filter) & set(r.get("issues", [])):
                continue
            hits.append({"id": r.get("id"), "score": float(score), "text": r["text"], "risk": r.get("risk"),
                         "issues": r.get("issues", []), "meta": r.get("meta", {})})
            if len(hits) >= k:
                break
        return hits


def evaluate_retrieval(index: NoteIndex, queries: list[dict], k: int = 5) -> dict:
    """precision@k: share of the top-k hits that share an active issue with the query note."""
    precisions, same_risk = [], []
    for q in queries:
        if not q["issues"]:
            continue
        hits = index.search(q["text"], k=k, exclude_id=q.get("id"))
        precisions.append(sum(bool(set(h["issues"]) & set(q["issues"])) for h in hits) / k)
        same_risk.append(sum(h["risk"] == q["risk"] for h in hits) / k)
    return {"queries": len(precisions), f"precision@{k}": float(np.mean(precisions)),
            f"same_risk@{k}": float(np.mean(same_risk))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/test_unseen.jsonl")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval", action="store_true", help="compare embeddings vs TF-IDF retrieval")
    ap.add_argument("--query", default=None, help="free-text query to search for")
    args = ap.parse_args(argv)

    corpus = read_jsonl(args.corpus)
    if args.eval:
        queries = read_jsonl(args.queries)[: args.n_queries]
        for backend in ("tfidf", "embeddings"):
            try:
                index = NoteIndex(corpus, backend=backend)
            except ImportError as exc:
                print(f"{backend:<11} skipped ({exc})")
                continue
            res = evaluate_retrieval(index, queries, k=args.k)
            print(f"{backend:<11} {res}")
    if args.query:
        for h in NoteIndex(corpus).search(args.query, k=args.k):
            print(f"{h['score']:.3f}  {h['id']}  risk={h['risk']}  issues={h['issues']}")


if __name__ == "__main__":
    main()
