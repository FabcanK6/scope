"""Gemini API client and helpers for SCOPE's LLM engine (free tier friendly).

Gemini client and shared helpers. The note-reading engine itself is in
:mod:`scope.engine`; this module holds the transport, the severity rubric text,
quote verification, rubric scoring, and the follow-up letter.

The API key comes from the ``GEMINI_API_KEY`` environment variable or Streamlit
secrets. Nothing runs unless a key is configured. Free-tier requests may be
used by the provider, so only send fictional or de-identified notes.

    GEMINI_API_KEY=... python -m scope.llm --file note.txt
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import urllib.error
import urllib.request

from scope import profile as _profile

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "")
# Gemini 3 models think before answering; their default ("medium") can take over a minute on a long note. Reading a
# note is careful extraction, not puzzle solving, and SCOPE's code does the scoring and date counting, so "low" keeps
# answers fast. Override with GEMINI_THINKING (low / medium / high).
THINKING_LEVEL = os.environ.get("GEMINI_THINKING", "low")
STATUSES = ["active", "resolved_on_site", "no_issue"]
SEVERITIES = ["minor", "major", "critical"]


BUSY_MESSAGE = ("Gemini is busy right now (Google's free tier is under high demand). "
                "Please try again in a minute.")


class LLMError(Exception):
    """A user-facing error message (quota reached, bad key, network problem...)."""


class ModelNotFound(LLMError):
    """This model can't be used with this key; try another one."""


class QuotaExceeded(LLMError):
    """This model's free quota is used up (per minute or per day); other models have their own quota."""

    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.per_day = bool(re.search(r"PerDay", detail))
        m = re.search(r'"retryDelay":\s*"(\d+)', detail)
        self.retry_seconds = int(m.group(1)) if m else None


def quota_message(e: QuotaExceeded) -> str:
    if e.per_day:
        return ("Today's free Gemini quota is used up for every model SCOPE can use. It resets at midnight Pacific "
                "time. Notes that were already read today still open from the cache.")
    wait = f"about {e.retry_seconds} seconds" if e.retry_seconds else "a minute"
    return f"Gemini's free per-minute limit was reached. Wait {wait} and try again."


class BadAnswer(LLMError):
    """The model answered, but not with usable JSON. ``raw`` keeps the answer for diagnosis."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class ModelBusy(LLMError):
    """Temporary overload (HTTP 500/502/503/504 or a timeout); retry, then try another model."""


class ModelSlow(ModelBusy):
    """The model did not answer within the timeout: go straight to the next model (waiting again rarely helps)."""


# ---------------------------------------------------------------------------
# Gemini REST client (standard library only)
# ---------------------------------------------------------------------------
class LLMClient:
    """What SCOPE needs from any LLM provider: ``generate`` (text, optionally constrained to a JSON schema written in
    SCOPE's schema dialect) plus a ``model`` name. Provider classes implement ``generate``."""

    provider = "LLM"
    model: str | None = None

    def __init__(self) -> None:
        self.avoid: set[str] = set()  # models that just gave a bad answer (used by providers with several models)
        self.trail: list[str] = []  # what happened on each try, e.g. "gemini-3.8-flash: timed out after 60 s"

    @property
    def label(self) -> str:
        return f"{self.provider} {self.model or ''}".strip()

    def generate(self, system: str, prompt: str, schema: dict | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        raw = self.generate(system, prompt, schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            try:
                if m:
                    return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
            raise BadAnswer("Gemini did not return valid JSON.", raw) from None


class GeminiClient(LLMClient):
    provider = "Google Gemini"
    retry_delays = (2.0, 5.0)  # waits before the 2nd and 3rd attempt on a busy model
    max_models = 6  # how many models to try before giving up
    time_budget = 150.0  # seconds: stop trying further models after this, so nobody waits for many minutes
    slow_for = 600.0  # seconds a model that timed out is tried last

    def __init__(self, api_key: str, model: str | None = None, timeout: int = 60):
        super().__init__()
        if not api_key:
            raise LLMError("No Gemini API key configured.")
        self.api_key = api_key
        self.preferred = model or DEFAULT_MODEL or None  # pinned via GEMINI_MODEL, tried first
        self.model = self.preferred  # the last model that answered
        self.timeout = timeout
        self._listed: list[str] | None = None
        self._sleep = time.sleep
        self._clock = time.monotonic
        self._slow: dict[str, float] = {}  # model -> when it last timed out
        self._no_thinking: set[str] = set()  # models that refused thinkingConfig

    # -- transport (patched in tests) -------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{GEMINI_BASE}/{path}", method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:3000]
            if e.code == 429 and re.search(r"limit:\s*0\b", detail):
                raise ModelNotFound(f"model not available on this tier: {detail}") from e  # try another model
            if e.code == 429:
                raise QuotaExceeded(detail) from e  # try another model: free quotas are per model
            if e.code in (401, 403):
                raise LLMError("The Gemini API key was rejected. Check the key in the app settings.") from e
            if e.code == 404:
                raise ModelNotFound(detail) from e
            if e.code in (500, 502, 503, 504):
                raise ModelBusy(f"HTTP {e.code}") from e
            raise LLMError(f"Gemini API error {e.code}: {_message(detail)}") from e
        except TimeoutError as e:
            raise ModelSlow("timeout") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ModelSlow("timeout") from e
            raise LLMError(f"Could not reach the Gemini API ({e.reason}).") from e

    # -- model selection --------------------------------------------------
    def list_models(self) -> list[str]:
        data = self._request("GET", "models?pageSize=200")
        names = []
        for m in data.get("models", []):
            name = m.get("name", "").removeprefix("models/")
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if "flash" in name and not re.search(r"tts|image|live|audio|embed|thinking|exp", name):
                names.append(name)

        def key(n: str):
            version = [int(x) for x in re.findall(r"\d+", n)[:2]] or [0]
            return (0 if "preview" in n else 1, version, 0 if "lite" in n else 1)

        return sorted(names, key=key, reverse=True)

    def candidates(self) -> list[str]:
        """Last working model, then the pinned one, then the newest available flash models."""
        if self._listed is None:
            try:
                self._listed = self.list_models()
            except ModelBusy:
                self._listed = []
        order = list(dict.fromkeys(m for m in [self.model, self.preferred, *self._listed, "gemini-flash-latest"] if m))
        now = self._clock()
        slow = {m for m, t in self._slow.items() if now - t < self.slow_for}
        last = self.avoid | slow
        order = [m for m in order if m not in last] + [m for m in order if m in last]
        return order[: self.max_models]

    def _body_for(self, model: str, body: dict) -> dict:
        """Gemini 3 models get a low thinking level (see THINKING_LEVEL); older models keep their defaults."""
        if not re.match(r"gemini-([3-9]|\d\d)", model) or model in self._no_thinking or not THINKING_LEVEL:
            return body
        config = {**body["generationConfig"], "thinkingConfig": {"thinkingLevel": THINKING_LEVEL}}
        return {**body, "generationConfig": config}

    # -- generation -------------------------------------------------------
    def generate(self, system: str, prompt: str, schema: dict | None = None) -> str:
        # No temperature: Google advises keeping Gemini 3 models at their default, because a low temperature can
        # cause looping and degraded answers (and newer Flash models ignore it).
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {},
        }
        if schema is not None:
            body["generationConfig"].update(responseMimeType="application/json", responseSchema=schema)
        last = None
        started = self._clock()
        for model in self.candidates():
            if last is not None and self._clock() - started > self.time_budget:
                break  # tried long enough: say so instead of keeping the user waiting
            t0 = self._clock()
            try:
                try:
                    data = self._call_with_retry(f"models/{model}:generateContent", self._body_for(model, body))
                except LLMError as e:
                    if "thinking" not in str(e).lower() or model in self._no_thinking:
                        raise
                    self._no_thinking.add(model)  # this model does not take a thinking level: ask without it
                    data = self._call_with_retry(f"models/{model}:generateContent", body)
            except (ModelNotFound, ModelBusy, QuotaExceeded) as e:
                self.trail.append(f"{model}: {_outcome(e)} after {self._clock() - t0:.0f} s")
                if isinstance(e, ModelSlow):
                    self._slow[model] = self._clock()
                if not isinstance(last, QuotaExceeded):  # report the quota problem if any model hit it
                    last = e
                continue
            except LLMError as e:
                self.trail.append(f"{model}: error after {self._clock() - t0:.0f} s ({str(e)[:120]})")
                raise
            self.trail.append(f"{model}: answered in {self._clock() - t0:.0f} s")
            self.model = model  # remember the model that worked
            try:
                return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            except (KeyError, IndexError) as e:
                reason = data.get("promptFeedback", {}).get("blockReason") or "empty response"
                raise LLMError(f"Gemini returned no text ({reason}).") from e
        if isinstance(last, QuotaExceeded):
            raise LLMError(quota_message(last))
        if isinstance(last, ModelBusy):
            raise LLMError(BUSY_MESSAGE)
        raise LLMError(f"No usable Gemini model found ({last}).")

    def _call_with_retry(self, path: str, body: dict) -> dict:
        """Busy models get two more tries after a short wait; a timeout moves on at once."""
        for delay in (*self.retry_delays, None):
            try:
                return self._request("POST", path, body)
            except ModelBusy as e:
                if delay is None or isinstance(e, ModelSlow):
                    raise
                self._sleep(delay)
        raise AssertionError("unreachable")



# ---------------------------------------------------------------------------
# Second opinion
# ---------------------------------------------------------------------------
# The rubric now lives in study profiles (scope/profile.py); this is the default profile's text (rubric v3.2).
RUBRIC_TEXT = _profile.rubric_text(_profile.default_profile())
ESCALATION_SUBJECTS = 3  # default profile: a problem affecting this many subjects or more is raised one level


def final_severity(f: dict, profile: dict | None = None) -> tuple[str, list[str]]:
    """Escalation for one finding under a profile (default: rubric v3.2). See ``scope.profile.final_severity``."""
    return _profile.final_severity(f, profile or _profile.default_profile())


def _outcome(e: Exception) -> str:
    if isinstance(e, ModelSlow):
        return "timed out"
    if isinstance(e, QuotaExceeded):
        return "daily free quota used up" if e.per_day else "per-minute limit reached"
    if isinstance(e, ModelNotFound):
        return "not available with this key"
    return f"busy ({e})"


def _message(detail: str) -> str:
    """The human-readable part of a Gemini error body (falls back to the raw text)."""
    try:
        return json.loads(detail)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return detail


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.lower().split())


_MAX_GAP = 250  # characters allowed between the parts of a quote joined with "..."
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|\u2026|\[\.\.\.\])\s*")


def verify_quote(quote: str, note: str) -> bool:
    """True when the quote appears in the note (ignoring case, spacing and curly quotes),
    or matches a sentence of the note almost exactly.

    Models often join two parts of a note with an ellipsis ("... shows an episode ... The site did not report it",
    "SAEs ... was in order"). Such a quote counts when every part is in the note, in that order, close together
    (at most ``_MAX_GAP`` characters apart), and the parts are not trivially short."""
    parts = [p for p in _ELLIPSIS.split(quote or "") if p.strip(" .")]
    if len(parts) > 1:
        n, at = _norm(note), None
        qs = [_norm(part).strip(" .") for part in parts]
        if min(len(q) for q in qs) < 4 or sum(len(q) for q in qs) < 15:
            return False
        for q in qs:
            i = n.find(q, at or 0)
            if i < 0 or (at is not None and i - at > _MAX_GAP):
                return False
            at = i + len(q)
        return True
    q, n = _norm(quote).strip(" ."), _norm(note)
    if not q:
        return False
    if q in n:
        return True
    sentences = re.split(r"(?<=[.;!?])\s+|\n", note)
    return any(difflib.SequenceMatcher(None, q, _norm(s).strip(" .")).ratio() >= 0.9 for s in sentences if s.strip())


def score_findings(findings: list[dict], profile: dict | None = None) -> dict:
    """Apply a profile's rubric to verified, active findings (highest final severity per issue type)."""
    return _profile.score(findings, profile or _profile.default_profile())


# ---------------------------------------------------------------------------
# Follow-up letter
# ---------------------------------------------------------------------------
LETTER_SYSTEM = """You draft post-visit follow-up letters from the study's monitoring contact to a
clinical trial site's Principal Investigator. Use a professional, concise tone. Use only the facts provided; where a
detail is missing (names, dates, protocol number), write a placeholder in square brackets such as [Protocol number].
Structure: greeting; one-paragraph visit overview; "Findings requiring action" (numbered, each with what is needed
and the due date if known); "Resolved during the visit"; "Reminders"; closing with [Name] and [Title]
placeholders.
Never add findings that are not in the facts. Output plain Markdown."""


def draft_followup(client: LLMClient, record: dict) -> str:
    v = record["visit"]
    facts = {
        "visit_type": v["visit_type"]["name"], "visit_date": v["visit_date"]["iso"] or v["visit_date"]["text"],
        "site": v["site"]["text"], "monitor": v["monitor"], "pi": v["pi"], "risk_level": record["risk"]["level"],
        "findings": [{"display": f["display"], "status": f["status"], "evidence": f["evidence"],
                      "severity": f.get("final_severity", f["severity"]), "raised_because": f.get("escalated_by", [])}
                     for f in record.get("findings", []) if f.get("verified") and f["status"] != "no_issue"],
        "action_items": [{k: a[k] for k in ("action", "owner", "due")} for a in record["actions"]],
    }
    return client.generate(LETTER_SYSTEM, "Facts from the visit (JSON):\n" + json.dumps(facts, indent=2)).strip()


def get_api_key(secrets=None) -> str | None:
    """Environment variable first, then Streamlit secrets (if given)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key and secrets is not None:
        try:
            key = secrets.get("GEMINI_API_KEY")
        except Exception:  # no secrets file
            key = None
    return key or None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--letter", action="store_true", help="also draft the follow-up letter")
    args = ap.parse_args(argv)
    from scope.engine import LLMParser
    from scope.record import audit_summary

    note = open(args.file).read()
    parser = LLMParser(GeminiClient(get_api_key() or ""))
    rec = parser.analyze(note)
    print(audit_summary(rec))
    if args.letter:
        print("\n" + draft_followup(parser.client, rec))


if __name__ == "__main__":
    main()
