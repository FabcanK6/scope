"""Demo studies shipped with SCOPE (all fictional): a protocol PDF, a ready-made study profile and example notes.

Built by ``scripts/make_demo_studies.py`` into ``profiles/demos/<study>/``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from scope import profile as P

DEMO_DIR = Path(__file__).resolve().parents[1] / "profiles" / "demos"


@lru_cache(maxsize=1)
def index() -> list[dict]:
    path = DEMO_DIR / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


def profile(study_id: str) -> dict:
    return P.loads((DEMO_DIR / study_id / "profile.json").read_text())


@lru_cache(maxsize=16)
def notes(study_id: str) -> tuple[dict, ...]:
    return tuple(json.loads((DEMO_DIR / study_id / "notes.json").read_text()))


def protocol_path(study_id: str) -> Path:
    return DEMO_DIR / study_id / "protocol.pdf"


def labelled_rows() -> list[dict]:
    """Every demo note as an accuracy-check row, with the study it must be scored under."""
    return [{"id": n["id"], "text": n["text"], "risk": n["expected_risk"], "issues": [], "risk_only": True,
             "study": s["id"], "why": n["why"]} for s in index() for n in notes(s["id"])]
