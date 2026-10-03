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

from scope.schema import SEVERITY_POINTS, risk_from_points

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "")
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


# ---------------------------------------------------------------------------
# Gemini REST client (standard library only)
# ---------------------------------------------------------------------------
class GeminiClient:
    retry_delays = (2.0, 5.0)  # waits before the 2nd and 3rd attempt on a busy model
    max_models = 6  # how many models to try before giving up

    def __init__(self, api_key: str, model: str | None = None, timeout: int = 60):
        if not api_key:
            raise LLMError("No Gemini API key configured.")
        self.api_key = api_key
        self.preferred = model or DEFAULT_MODEL or None  # pinned via GEMINI_MODEL, tried first
        self.model = self.preferred  # the last model that answered
        self.timeout = timeout
        self._listed: list[str] | None = None
        self._sleep = time.sleep
        self.avoid: set[str] = set()  # models that just gave a bad answer: tried last

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
            raise ModelBusy("timeout") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ModelBusy("timeout") from e
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
        order = [m for m in order if m not in self.avoid] + [m for m in order if m in self.avoid]
        return order[: self.max_models]

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
        for model in self.candidates():
            try:
                data = self._call_with_retry(f"models/{model}:generateContent", body)
            except (ModelNotFound, ModelBusy, QuotaExceeded) as e:
                if not isinstance(last, QuotaExceeded):  # report the quota problem if any model hit it
                    last = e
                continue
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
        for delay in (*self.retry_delays, None):
            try:
                return self._request("POST", path, body)
            except ModelBusy:
                if delay is None:
                    raise
                self._sleep(delay)
        raise AssertionError("unreachable")

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


# ---------------------------------------------------------------------------
# Second opinion
# ---------------------------------------------------------------------------
RUBRIC_TEXT = """Severity rubric v3 (approved by an experienced clinical research associate).
Rate each finding on its own facts; repeats and spread are recorded separately (see the escalation fields)
and SCOPE applies those itself.
- SAE_REPORTING. minor: SAE form detail wrong, corrected. major: SAE follow-up report overdue; PI causality not
  documented. critical: SAE unreported, or reported outside 24 hours (stays critical even with a CAPA).
- AE_REPORTING. minor: one AE entered late. major: AEs missing from EDC; grading or causality not assessed by the PI.
- CONSENT. minor: missing time of signature; initials missing on a page. major: outdated ICF version used (major even
  if the subject was re-consented during the visit); re-consent overdue. critical: study procedures before consent;
  no signed ICF.
- ELIGIBILITY. minor: eligibility checklist unsigned but criteria met. major: eligibility evidence missing from source
  at randomization. critical: ineligible subject randomized or dosed.
- SAFETY_REPORTS. minor: IND safety reports filed late in the ISF. major: safety reports not reviewed by the PI or not
  sent to the IRB.
- UNBLINDING. major: blinded staff could see unblinded documents, no unblinding. critical: unplanned unblinding not
  reported.
- PROTOCOL_DEVIATION. minor: a single out-of-window visit. major: important deviations; missed safety assessments;
  deviations not logged.
- DOSING_ERROR. minor: dosing time not recorded. major: missed doses undocumented; compliance not reconciled
  (including returned doses not counted).
  critical: wrong dose, double dose, or dosed despite a hold criterion.
- IP_ACCOUNTABILITY. minor: small count difference explained on site. major: kits unaccounted for; wrong kit
  dispensed. critical: expired investigational product dispensed.
- TEMP_EXCURSION. minor: brief excursion with no product impact, reported. major: excursion not reported; logs not
  kept. critical: product used after an excursion before Sponsor assessment.
- LAB_SAMPLES. minor: lab kit supplies running low. major: samples mishandled or not shipped; central lab results not
  reviewed.
- DATA_ENTRY_BACKLOG. minor: a few pages or one concomitant medication not entered. major: backlog older than 60 days,
  or a large volume (about 25 or more pages not entered).
- QUERY_AGING. minor: a few queries open. major: many queries open more than 60 days, or 10 or more queries older
  than 30 days.
- SDV_BACKLOG. minor: SDV slightly behind plan. major: SDV far behind; the monitor's EMR access lapsed. critical: the
  site refuses access to source documents.
- SOURCE_DOCS. minor: corrections not initialed or dated. major: source missing or contradicts EDC; ALCOA+ failures.
  critical: falsified or back-dated records.
- STAFF_TURNOVER (staff, training and delegation). minor: one CV or GCP certificate expired. major: staff not on the
  delegation log; turnover with no backup. critical: untrained or undelegated staff performing study procedures.
- PI_OVERSIGHT. minor: one late sign-off. major: PI not signing labs or eCRFs; PI unavailable to the team.
- ENROLLMENT_LAG. minor: slightly behind target. major: far behind target.
- REG_DOCS. minor: one document misfiled. major: missing 1572, amendment approval or licenses; a pending IRB approval
  that blocks screening. critical: enrolling after IRB approval lapsed.
- FACILITY_EQUIPMENT. minor: calibration due soon. major: equipment out of calibration; lab certification (CLIA/CAP)
  expired.
- PRIOR_ACTIONS (follow-up of prior findings). minor: one prior action slightly overdue. major: prior actions not done;
  CAPA not implemented.
- SITE_ENGAGEMENT. minor: slow replies to the monitor. major: unresponsive for weeks; repeated visit cancellations.
Status: "active" = a problem that still exists after the visit. "resolved_on_site" = it was corrected and verified
during the visit. "no_issue" = the topic is mentioned only to confirm it is fine. Critical findings stay "active" even
when a CAPA is in place.
Escalation fields (facts only, SCOPE does the scoring): "repeat" = true only when the note says this problem was also
found at an earlier visit or an earlier action on it is still open. "subjects_affected" = how many subjects the note
says the problem affects (null if not stated). "site_wide" = true when the note describes it as site-wide or
systemic. "escalation_evidence" = the words from the note that show the repeat or the spread ("" if neither)."""

ESCALATION_SUBJECTS = 3  # a problem affecting this many subjects or more is raised one level (up to major)


def final_severity(f: dict) -> tuple[str, list[str]]:
    """Rubric v3 escalation: many subjects (3+ or site-wide) raises one level up to major; a repeat finding raises
    one level. Both can apply. Only counted when the escalation evidence is really in the note (``escalation_ok``)."""
    level = SEVERITIES.index(f["severity"])
    reasons = []
    if f.get("escalation_ok"):
        n = f.get("subjects_affected")
        if (isinstance(n, int) and n >= ESCALATION_SUBJECTS) or f.get("site_wide"):
            if level < SEVERITIES.index("major"):
                level += 1
                reasons.append(f"{n} subjects affected" if isinstance(n, int) and n >= ESCALATION_SUBJECTS
                               else "site-wide")
        if f.get("repeat") and level < len(SEVERITIES) - 1:
            level += 1
            reasons.append("repeat finding")
    return SEVERITIES[level], reasons


def _message(detail: str) -> str:
    """The human-readable part of a Gemini error body (falls back to the raw text)."""
    try:
        return json.loads(detail)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return detail


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.lower().split())


def verify_quote(quote: str, note: str) -> bool:
    """True when the quote appears in the note (ignoring case, spacing and curly quotes),
    or matches a sentence of the note almost exactly."""
    q, n = _norm(quote).strip(" ."), _norm(note)
    if not q:
        return False
    if q in n:
        return True
    sentences = re.split(r"(?<=[.;!?])\s+|\n", note)
    return any(difflib.SequenceMatcher(None, q, _norm(s).strip(" .")).ratio() >= 0.9 for s in sentences if s.strip())


def score_findings(findings: list[dict]) -> dict:
    """Apply SCOPE's rubric to verified, active findings (highest final severity per issue type)."""
    worst: dict[str, str] = {}
    for f in findings:
        if f["status"] != "active" or not f.get("verified"):
            continue
        sev = f.get("final_severity", f["severity"])
        if f["issue"] not in worst or SEVERITY_POINTS[sev] > SEVERITY_POINTS[worst[f["issue"]]]:
            worst[f["issue"]] = sev
    points = sum(SEVERITY_POINTS[s] for s in worst.values())
    return {"risk": risk_from_points(points), "points": points, "active": worst}


# ---------------------------------------------------------------------------
# Follow-up letter
# ---------------------------------------------------------------------------
LETTER_SYSTEM = """You draft post-visit follow-up letters from a clinical research associate (CRA) to a clinical
trial site's Principal Investigator. Use a professional, concise tone. Use only the facts provided; where a detail
is missing (names, dates, protocol number), write a placeholder in square brackets such as [Protocol number].
Structure: greeting; one-paragraph visit overview; "Findings requiring action" (numbered, each with what is needed
and the due date if known); "Resolved during the visit"; "Reminders"; closing with the CRA's name placeholder.
Never add findings that are not in the facts. Output plain Markdown."""


def draft_followup(client: GeminiClient, record: dict) -> str:
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
