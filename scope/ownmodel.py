"""SCOPE's own model: a small text model trained on expert-labelled notes and on the cases reviewers approve.

It is the second way SCOPE learns (the first is the shared rulings in :mod:`scope.learning`). It retrains itself in
seconds whenever the approved cases change, measures itself on the expert-labelled notes it was not trained on, and
only switches itself on when it clears the bar (``ACTIVATION_BAR``). Until then it stays in the background and its
score is shown, so everyone can watch it improve. Once on, it gives a second opinion on each visit's risk level and
a rough estimate when the AI model is unavailable.

No new SCOPE version is needed for it to improve: more approved cases mean a better model, automatically.
"""

from __future__ import annotations

import datetime as _dt
import hashlib

import numpy as np

RISKS = ["low", "medium", "high"]
ACTIVATION_BAR = 0.80  # agreement with expert risk levels on notes it has not seen
HIGH_RECALL_BAR = 0.80  # and it must catch this share of high-risk visits
FOLDS, SEEDS = 5, (0, 1, 2)


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def expert_rows() -> list[dict]:
    """Notes with expert risk labels (hand-written, formal reports, stress set). Demo notes are left out: their risk
    depends on each study's protocol, which this model does not see."""
    from scope.data.handwritten import load_handwritten, load_realistic, load_stress

    return [{"text": r["text"], "risk": r["risk"], "source": "expert", "id": r["id"]}
            for r in load_handwritten() + load_realistic() + load_stress() if r.get("risk") in RISKS]


def dataset(library: dict | None = None, experts: list[dict] | None = None,
            practice: list[dict] | None = None) -> list[dict]:
    """Expert notes, practice notes the community agreed on (both also used to test the model), and approved
    shared cases (training only)."""
    rows = list(experts if experts is not None else expert_rows())
    if library:
        from scope import learning as L

        if practice is None:
            from scope.data.handwritten import load_practice

            practice = load_practice()
        rows += [{"text": r["text"], "risk": r["risk"], "source": "community", "id": r["id"]}
                 for r in L.community_rows(library, practice)]
        rows += [{"text": r["text"], "risk": r["risk"], "source": "shared", "id": r["id"]}
                 for r in L.training_rows(library) if r.get("risk") in RISKS]
    return rows


TESTED = ("expert", "community")  # labelled by people independently of SCOPE: fair to test on


def _pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(TfidfVectorizer(sublinear_tf=True, stop_words="english"),
                         LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced"))


def evaluate(rows: list[dict]) -> dict:
    """Agreement with the experts on expert notes the model was not trained on (5-fold, repeated 3 times). Shared
    cases are always in the training folds, except a shared copy of a note being tested."""
    from sklearn.model_selection import StratifiedKFold

    experts = [r for r in rows if r["source"] in TESTED]
    shared = [r for r in rows if r["source"] not in TESTED]
    y = np.array([r["risk"] for r in experts])
    if len(experts) < FOLDS * 2 or min(np.sum(y == k) for k in set(y)) < FOLDS:
        return {"accuracy": None, "high_recall": None, "tested": 0}
    right, high_hit, high_n = 0, 0, 0
    for seed in SEEDS:
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            test_texts = {_norm(experts[i]["text"]) for i in te}
            train = [experts[i] for i in tr] + [r for r in shared if _norm(r["text"]) not in test_texts]
            model = _pipeline().fit([r["text"] for r in train], [r["risk"] for r in train])
            pred = model.predict([experts[i]["text"] for i in te])
            truth = y[te]
            right += int(np.sum(pred == truth))
            high_n += int(np.sum(truth == "high"))
            high_hit += int(np.sum((pred == "high") & (truth == "high")))
    n = len(experts) * len(SEEDS)
    return {"accuracy": right / n, "high_recall": high_hit / high_n if high_n else None, "tested": len(experts)}


class OwnModel:
    """The trained model plus its report card."""

    def __init__(self, pipeline, report: dict):
        self.pipeline, self.report = pipeline, report

    @property
    def active(self) -> bool:
        return bool(self.report.get("active"))

    def predict(self, text: str) -> dict:
        probs = self.pipeline.predict_proba([text])[0]
        by_risk = {str(c): float(p) for c, p in zip(self.pipeline.classes_, probs)}
        risk = max(by_risk, key=by_risk.get)
        return {"risk": risk, "confidence": by_risk[risk], "probs": by_risk}


def train(library: dict | None = None, experts: list[dict] | None = None, bar: float = ACTIVATION_BAR,
          practice: list[dict] | None = None) -> OwnModel:
    rows = dataset(library, experts, practice)
    score = evaluate(rows)
    pipeline = _pipeline().fit([r["text"] for r in rows], [r["risk"] for r in rows])
    acc, rec = score["accuracy"], score["high_recall"]
    report = {
        **score, "bar": bar, "high_recall_bar": HIGH_RECALL_BAR,
        "expert_notes": sum(r["source"] == "expert" for r in rows),
        "community_notes": sum(r["source"] == "community" for r in rows),
        "shared_cases": sum(r["source"] == "shared" for r in rows),
        "trained": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "data": hashlib.sha256("|".join(sorted(r["id"] for r in rows)).encode()).hexdigest()[:10],
        "active": acc is not None and acc >= bar and (rec is None or rec >= HIGH_RECALL_BAR),
    }
    return OwnModel(pipeline, report)


def second_opinion(model: OwnModel | None, text: str, ai_risk: str) -> dict | None:
    """The model's view when it is switched on, and whether it calls for a second look: it reads the visit as high
    risk while the AI reading says low (the costly miss)."""
    if model is None or not model.active:
        return None
    guess = model.predict(text)
    guess["second_look"] = guess["risk"] == "high" and ai_risk == "low"
    return guess
