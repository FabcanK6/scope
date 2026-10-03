"""Shared learning: SCOPE gets better from the people who use it.

When someone corrects a reading (or confirms SCOPE got it right) and chooses to share it, the case goes into a
shared library as *pending*. A curator reviews it; once *approved*, SCOPE shows the ruling to the AI model whenever
it reads a similar note, for every user, from that moment on. No new version of SCOPE is needed for it to learn.

* Corrections become rulings in the prompt (retrieved by similarity, like a reviewer remembering similar cases).
* Confirmations and corrections together are the training data for SCOPE's own model (the next layer).
* Nothing changes SCOPE's behaviour without a person approving it, and every case keeps who, when and why: the audit
  trail a sponsor would ask for.

Storage is a folder of small JSON files (``pending/``, ``approved/``, ``rejected/``), either local (tests, running
on your own computer) or a private Hugging Face dataset repo (the public app). Only fictional or de-identified notes
may be shared.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import uuid
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FOLDERS = ("pending", "approved", "rejected")
KINDS = ("correction", "confirmation")
ALL_STUDIES = "all"
MIN_SIMILARITY = 0.12  # rulings less similar than this to the note are not shown to the model
STATUS_WORDS = {"active": "an active problem", "resolved_on_site": "fixed during the visit",
                "no_issue": "not a problem"}


class LearningError(Exception):
    """The shared library could not be read or written (SCOPE keeps working without it)."""


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _norm_note(text: str) -> str:
    return " ".join((text or "").lower().split())


def note_hash(text: str) -> str:
    return hashlib.sha256(_norm_note(text).encode()).hexdigest()[:16]


def make_case(kind: str, note: str, rec: dict, profile: dict, correction: dict | None = None, by: str = "") -> dict:
    """A shareable case from a reading. ``correction`` (for kind 'correction'): issue, display, status, severity,
    quote, reason, risk."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    if kind == "correction" and not correction:
        raise ValueError("a correction needs its details")
    protocol = (profile.get("protocol") or {}).get("reference")
    found = [f for f in rec.get("findings", []) if f.get("verified")]
    return {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "by": by.strip()[:60],
        "note": note,
        "note_hash": note_hash(note),
        "study": {"name": profile.get("name", ""), "version": profile.get("version", ""), "protocol": protocol},
        "scope_said": {"risk": (rec.get("risk") or {}).get("level"),
                       "findings": [{"issue": f["issue"], "status": f["status"],
                                     "severity": f.get("final_severity", f.get("severity")),
                                     "evidence": f.get("evidence", "")} for f in found]},
        "correction": correction if kind == "correction" else None,
        "engine": {"provider": rec.get("provider"), "model": rec.get("model")},
        "status": "pending",
        "applies_to": None,
        "decided": None,
        "curator_note": "",
    }


def validate_case(case: dict) -> dict:
    if not isinstance(case, dict) or case.get("kind") not in KINDS or not case.get("id"):
        raise LearningError("not a SCOPE case")
    if not re.fullmatch(r"[0-9a-f]{6,32}", str(case["id"])):
        raise LearningError("bad case id")
    if case["kind"] == "correction":
        c = case.get("correction") or {}
        if not c.get("issue") or c.get("status") not in STATUS_WORDS or not str(c.get("quote") or "").strip():
            raise LearningError("incomplete correction")
    return case


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
class Store:
    """Where shared cases live. ``load`` returns {'pending': [...], 'approved': [...], 'rejected': [...]}."""

    label = "store"

    def load(self) -> dict[str, list[dict]]:  # pragma: no cover
        raise NotImplementedError

    def add(self, case: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    def decide(self, case: dict, approve: bool, applies_to: str | None = None, note: str = "") -> dict:
        """Approve (for all studies or one study) or reject a pending case."""
        decided = {**case, "status": "approved" if approve else "rejected",
                   "applies_to": (applies_to or ALL_STUDIES) if approve else None,
                   "decided": _dt.date.today().isoformat(), "curator_note": note.strip()[:300]}
        self._move(case, decided)
        return decided

    def _move(self, old: dict, new: dict) -> None:  # pragma: no cover
        raise NotImplementedError


def _group(cases: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {f: [] for f in FOLDERS}
    for c in cases:
        try:
            validate_case(c)
        except LearningError:
            continue
        out.setdefault(c.get("status", "pending"), []).append(c)
    for v in out.values():
        v.sort(key=lambda c: c.get("created", ""))
    return out


class LocalStore(Store):
    """A folder on disk (tests, or SCOPE running on your own computer)."""

    label = "local folder"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self) -> dict[str, list[dict]]:
        cases = []
        for folder in FOLDERS:
            for p in sorted((self.root / folder).glob("*.json")):
                try:
                    cases.append({**json.loads(p.read_text()), "status": folder})
                except ValueError:
                    continue
        return _group(cases)

    def add(self, case: dict) -> None:
        validate_case(case)
        path = self.root / "pending" / f"{case['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case, ensure_ascii=False, indent=1))

    def _move(self, old: dict, new: dict) -> None:
        src = self.root / old.get("status", "pending") / f"{old['id']}.json"
        dst = self.root / new["status"] / f"{new['id']}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(new, ensure_ascii=False, indent=1))
        if src.exists() and src != dst:
            src.unlink()


class HFStore(Store):
    """A private Hugging Face dataset repo (free). Each case is one small JSON file, so writes never clash."""

    label = "Hugging Face dataset"

    def __init__(self, repo_id: str, token: str):
        if not repo_id or "/" not in repo_id or not token:
            raise LearningError("Shared learning needs SCOPE_LEARNING_REPO (user/name) and HF_TOKEN.")
        self.repo_id, self.token = repo_id, token

    def _hub(self):
        try:
            import huggingface_hub
        except ImportError as e:  # pragma: no cover - installed from requirements.txt on the app
            raise LearningError("The huggingface_hub package is not installed.") from e
        return huggingface_hub

    def load(self) -> dict[str, list[dict]]:
        hub = self._hub()
        try:
            local = hub.snapshot_download(repo_id=self.repo_id, repo_type="dataset", token=self.token,
                                          allow_patterns=[f"{f}/*.json" for f in FOLDERS])
        except Exception as e:
            raise LearningError(f"Could not read the shared library ({type(e).__name__}).") from e
        return LocalStore(local).load()

    def _commit(self, operations: list, message: str) -> None:
        hub = self._hub()
        try:
            hub.HfApi(token=self.token).create_commit(repo_id=self.repo_id, repo_type="dataset",
                                                      operations=operations, commit_message=message)
        except Exception as e:
            raise LearningError(f"Could not save to the shared library ({type(e).__name__}).") from e

    def add(self, case: dict) -> None:
        validate_case(case)
        hub = self._hub()
        data = json.dumps(case, ensure_ascii=False, indent=1).encode()
        self._commit([hub.CommitOperationAdd(path_in_repo=f"pending/{case['id']}.json", path_or_fileobj=data)],
                     f"New {case['kind']} {case['id']} (pending review)")

    def _move(self, old: dict, new: dict) -> None:
        hub = self._hub()
        data = json.dumps(new, ensure_ascii=False, indent=1).encode()
        ops = [hub.CommitOperationAdd(path_in_repo=f"{new['status']}/{new['id']}.json", path_or_fileobj=data)]
        if old.get("status", "pending") != new["status"]:
            ops.append(hub.CommitOperationDelete(path_in_repo=f"{old.get('status', 'pending')}/{old['id']}.json"))
        self._commit(ops, f"{new['status'].capitalize()} {new['kind']} {new['id']}")


def digest(library: dict[str, list[dict]]) -> str:
    """Changes whenever the approved rulings change (part of the answer cache key)."""
    ids = sorted(f"{c['id']}:{c.get('applies_to')}" for c in library.get("approved", []))
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# using what was learned
# ---------------------------------------------------------------------------
def rulings(library: dict[str, list[dict]] | list[dict], study_name: str, topics: dict) -> list[dict]:
    """Approved corrections that apply to this study (all-study rulings plus this study's own) and use topics this
    study has switched on."""
    approved = library.get("approved", []) if isinstance(library, dict) else library
    out = []
    for c in approved:
        if c.get("kind") != "correction" or c.get("status", "approved") != "approved":
            continue
        if c.get("applies_to") not in (ALL_STUDIES, study_name):
            continue
        issue = (c.get("correction") or {}).get("issue")
        if issue in topics and topics[issue].get("enabled", True):
            out.append(c)
    return out


# words every visit note shares (headers, dates): they say nothing about which ruling applies
_STOP = sorted(set(ENGLISH_STOP_WORDS) | {
    "imv", "siv", "cov", "visit", "visits", "site", "sites", "monitoring", "interim", "report", "note", "subject",
    "subjects", "study", "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "cra"})


def _context(c: dict) -> str:
    """The ruling's quote (twice, it matters most) and the sentence around it in its note."""
    quote, note = c["correction"]["quote"], c.get("note", "")
    i = _norm_note(note).find(_norm_note(quote))
    around = ""
    if i >= 0:
        flat = _norm_note(note)
        start = max(flat.rfind(". ", 0, i), flat.rfind("; ", 0, i))
        end = min([j for j in (flat.find(". ", i), flat.find("; ", i)) if j >= 0] or [len(flat)])
        around = flat[start + 1:end]
    return f"{quote} {quote} {around}"


def relevant(cases: list[dict], note: str, k: int = 5, exclude_same_note: bool = False) -> list[tuple[dict, float]]:
    """The rulings most similar to this note (TF-IDF over each case's note and quote), above MIN_SIMILARITY.
    ``exclude_same_note`` keeps a note from seeing its own ruling (fair accuracy checks)."""
    if exclude_same_note:
        h = note_hash(note)
        cases = [c for c in cases if c.get("note_hash") != h]
    if not cases or not note.strip():
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [_context(c) for c in cases]
    vec = TfidfVectorizer(stop_words=_STOP, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", ngram_range=(1, 2),
                          sublinear_tf=True).fit(docs + [note])
    sims = cosine_similarity(vec.transform([note]), vec.transform(docs))[0]
    order = sorted(range(len(cases)), key=lambda i: -sims[i])
    return [(cases[i], float(sims[i])) for i in order[:k] if sims[i] >= MIN_SIMILARITY]


def ruling_line(c: dict, topics: dict) -> str:
    x = c["correction"]
    name = topics.get(x["issue"], {}).get("display", x.get("display") or x["issue"])
    verdict = STATUS_WORDS[x["status"]] + (f", {x['severity']}" if x["status"] == "active" and x.get("severity")
                                           else "")
    why = f" Reason: {x['reason']}" if x.get("reason") else ""
    return f"- When a note says \"{x['quote']}\": {x['issue']} ({name}) is {verdict}.{why}"


def prompt_text(picked: list[tuple[dict, float]], topics: dict) -> str:
    if not picked:
        return ""
    lines = ["Rulings from expert reviewers on similar notes (follow them when this note says something similar; "
             "this study's own rules above still come first):"]
    lines += [ruling_line(c, topics) for c, _ in picked]
    return "\n".join(lines)


def training_rows(library: dict[str, list[dict]]) -> list[dict]:
    """Approved cases as labelled examples for SCOPE's own model: the note, the expert's risk level and topics.
    A confirmation keeps SCOPE's reading; a correction changes the corrected topic (and the risk, if given)."""
    rows = []
    for c in library.get("approved", []):
        said = c.get("scope_said") or {}
        active = {f["issue"]: f["severity"] for f in said.get("findings", []) if f.get("status") == "active"}
        risk = said.get("risk")
        if c["kind"] == "correction":
            x = c["correction"]
            active.pop(x["issue"], None)
            if x["status"] == "active":
                active[x["issue"]] = x.get("severity") or "minor"
            risk = x.get("risk") or None  # without the expert's risk level the row is used for topics only
        rows.append({"id": c["id"], "text": c["note"], "risk": risk, "issues": sorted(active), "severities": active,
                     "study": (c.get("study") or {}).get("name"), "kind": c["kind"]})
    return rows
