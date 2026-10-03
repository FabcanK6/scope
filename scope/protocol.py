"""Protocol intake: read a study protocol and draft the study-specific rules SCOPE should score by.

    protocol (PDF / Word / text) ─► pages ─► the sections a monitor needs (safety reporting, AE/SAE definitions,
    visit windows, eligibility, dosing, IP storage, key procedures, consent, deviations)
    ─► LLM drafts rules, each with a word-for-word quote and page ─► SCOPE checks every quote against the protocol
    ─► a lead CRA accepts the rules ─► they become the study profile's rules, with the protocol cited

Nothing from the protocol is used until a person accepts it. Free-tier LLM requests may be used by the provider, so
only send protocols that are public (e.g. posted on ClinicalTrials.gov) or fictional.
"""

from __future__ import annotations

import copy
import hashlib
import io

from scope import profile as P
from scope.llm import GeminiClient, LLMError, verify_quote

GENERAL = "GENERAL"
SEVERITY_OPTIONS = ["minor", "major", "critical", "definition"]  # definition = changes what counts, not a severity

# Sections that matter for judging visit notes; used to pick pages when a protocol is too long to send whole.
KEYWORDS = {
    "serious adverse": 6, "sae": 4, "adverse event": 3, "report": 1, "business day": 6, "hours of": 3,
    "awareness": 3, "expedited": 3, "pregnan": 3, "special interest": 4, "aesi": 4,
    "visit window": 6, "window": 2, "schedule of": 3, "± ": 2, "+/-": 2,
    "inclusion criteria": 5, "exclusion criteria": 5, "eligib": 3,
    "dose": 2, "dosing": 3, "interrupt": 3, "discontinu": 2, "hold": 2, "modification": 2,
    "storage": 3, "temperature": 4, "excursion": 5, "°c": 2, "accountability": 3,
    "informed consent": 4, "re-consent": 4, "primary endpoint": 4, "pharmacokinetic": 3, "pk sample": 4,
    "protocol deviation": 6, "important deviation": 6, "major deviation": 5, "unblind": 4,
}
MAX_CHARS = 180_000  # about 45k tokens: fits the free tier comfortably

PROTOCOL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "study": {"type": "OBJECT", "properties": {
            k: {"type": "STRING", "nullable": True}
            for k in ("title", "protocol_number", "version", "phase", "therapeutic_area")}},
        "rules": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "topic": {"type": "STRING"},
            "rule": {"type": "STRING"},
            "severity": {"type": "STRING", "enum": SEVERITY_OPTIONS},
            "quote": {"type": "STRING"},
            "page": {"type": "INTEGER", "nullable": True}},
            "required": ["topic", "rule", "severity", "quote", "page"],
            "propertyOrdering": ["topic", "rule", "severity", "quote", "page"]}},
    },
    "required": ["study", "rules"],
    "propertyOrdering": ["study", "rules"],
}


def system_prompt(profile: dict) -> str:
    topics = "\n".join(f"- {t['code']}: {t['display']}" for t in P.enabled_topics(profile))
    return f"""You read clinical trial protocols for a clinical research associate (CRA). Extract the study-specific
rules a monitor needs to judge site visit notes for this study, so that a note is scored against THIS protocol rather
than general practice.

Look for, at least:
- SAE: how the protocol defines an SAE, events it says are NOT SAEs or not AEs (e.g. disease progression, elective or
  pre-planned hospitalizations), the reporting deadline (hours, calendar days or business days, from what moment) and
  what late reporting counts as.
- Other expedited reporting: AEs of special interest, pregnancy, overdose; their deadlines.
- Visit windows (per visit or general), and which visits or assessments are key (primary endpoint, PK samples, scans).
- Eligibility criteria that are often violated or safety-critical.
- Dosing: hold, stop and modification rules; compliance requirements.
- Investigational product: storage temperature range, what to do after an excursion, accountability rules.
- Consent and re-consent requirements; what the protocol lists as important protocol deviations.

Issue types in SCOPE (use one as "topic", or GENERAL if none fits):
{topics}
- GENERAL: applies to the whole study

The default severity rubric SCOPE uses when the protocol says nothing:
{P.rubric_text(profile)}

For each rule:
- "rule": one plain sentence written as an instruction for judging visit notes, with the concrete numbers from the
  protocol. Example: "SAEs must be reported to the sponsor within 2 business days of site awareness; later reporting is
  an important protocol deviation."
- "severity": what breaking the rule counts as. Use the protocol's own words where it gives them (an "important" or
  "major" deviation is major unless it involves subject safety or a missed SAE, which is critical); otherwise use the
  default rubric. Use "definition" for rules that change what counts (e.g. "Disease progression is not an SAE").
- "quote": the shortest sentence copied word for word from the protocol that shows the rule.
- "page": the number from the nearest [Page N] marker before the quote.
Only include rules that make the general rubric concrete or differ from it. Do not invent anything. At most 25 rules.
Also fill "study" with the title, protocol number, version or amendment, phase and therapeutic area if stated."""


# ---------------------------------------------------------------------------
# reading documents
# ---------------------------------------------------------------------------
def read_document(filename: str, data: bytes) -> list[str]:
    """Text of a protocol as a list of pages (PDF pages; Word and text files are cut into ~3,000-character pages)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise LLMError("Reading PDFs needs the pypdf package.") from e
        pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]
        if not "".join(pages).strip():
            raise LLMError("This PDF has no text layer (it may be a scan). Upload a text PDF or a Word file.")
        return pages
    if name.endswith(".docx"):
        try:
            import docx
        except ImportError as e:  # pragma: no cover
            raise LLMError("Reading Word files needs the python-docx package.") from e
        doc = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            text += "\n" + "\n".join(" | ".join(c.text for c in row.cells) for row in table.rows)
        return _paginate(text)
    if name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
        return [p for p in text.split("\f")] if "\f" in text else _paginate(text)
    raise LLMError("Upload the protocol as a PDF, Word (.docx) or text file.")


def _paginate(text: str, size: int = 3000) -> list[str]:
    pages, current = [], ""
    paras = []
    for para in text.split("\n"):  # very long lines are cut at word boundaries
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            paras.append(para[:cut])
            para = para[cut:].lstrip()
        paras.append(para)
    for para in paras:
        if len(current) + len(para) > size and current:
            pages.append(current)
            current = ""
        current += para + "\n"
    if current.strip():
        pages.append(current)
    return pages


def select_pages(pages: list[str], max_chars: int = MAX_CHARS) -> list[int]:
    """Indexes of the pages to send: all of them if they fit, else the pages that score highest on the sections a
    monitor needs (kept in document order)."""
    if sum(len(p) for p in pages) <= max_chars:
        return list(range(len(pages)))

    def score(text: str) -> float:
        low = text.lower()
        return sum(w * low.count(k) for k, w in KEYWORDS.items()) / (1 + len(text) / 3000)

    ranked = sorted(range(len(pages)), key=lambda i: -score(pages[i]))
    keep, total = [], 0
    for i in ranked:
        if total + len(pages[i]) > max_chars:
            continue
        keep.append(i)
        total += len(pages[i])
    return sorted(keep)


def marked_text(pages: list[str], keep: list[int]) -> str:
    return "\n\n".join(f"[Page {i + 1}]\n{pages[i].strip()}" for i in keep)


# ---------------------------------------------------------------------------
# drafting rules
# ---------------------------------------------------------------------------
def _page_of(quote: str, pages: list[str]) -> int | None:
    for i, page in enumerate(pages):
        if verify_quote(quote, page):
            return i + 1
    return None


def draft_rules(client: GeminiClient, pages: list[str], profile: dict | None = None) -> dict:
    """Ask the LLM for the study's rules and check every quote against the protocol text.

    Returns {"study": {...}, "rules": [...], "pages_read": n, "pages_total": n, "dropped": n}; each rule has
    topic, rule, severity, quote, page and verified=True (rules whose quote is not in the protocol are dropped)."""
    profile = profile or P.default_profile()
    keep = select_pages(pages)
    data = client.generate_json(system_prompt(profile), "Protocol:\n" + marked_text(pages, keep), PROTOCOL_SCHEMA)
    codes = {t["code"] for t in P.enabled_topics(profile)}
    rules, dropped = [], 0
    for r in data.get("rules") or []:
        quote = str(r.get("quote") or "").strip()
        page = _page_of(quote, pages) if quote else None
        if page is None:
            dropped += 1
            continue
        topic = str(r.get("topic") or GENERAL).upper()
        rules.append({"topic": topic if topic in codes else GENERAL, "rule": str(r.get("rule") or "").strip(),
                      "severity": r.get("severity") if r.get("severity") in SEVERITY_OPTIONS else "definition",
                      "quote": quote, "page": page})
    rules = [r for r in rules if r["rule"]]
    study = {k: (v or None) for k, v in (data.get("study") or {}).items()}
    return {"study": study, "rules": rules, "pages_read": len(keep), "pages_total": len(pages), "dropped": dropped}


def rule_line(r: dict, ref: str) -> str:
    """How an accepted protocol rule reads in the profile's study rules."""
    tags = [t for t in (r["topic"] if r["topic"] != GENERAL else "",
                        f"{r['severity']} if broken" if r["severity"] != "definition" else "") if t]
    tag = f" [{', '.join(tags)}]" if tags else ""
    return f"{r['rule']}{tag} (protocol {ref}, p. {r['page']})"


def apply_rules(profile: dict, draft: dict, accepted: list[int], filename: str, who: str = "",
                new_name: str | None = None) -> dict:
    """A new profile version with the accepted protocol rules added (and the protocol recorded)."""
    new = copy.deepcopy(profile)
    study = draft.get("study") or {}
    ref = study.get("protocol_number") or filename
    if study.get("version"):
        ref = f"{ref} {study['version']}"
    chosen = [draft["rules"][i] for i in accepted]
    new["study_rules"] = [*new["study_rules"], *(rule_line(r, ref) for r in chosen)]
    new["protocol"] = {"file": filename, "reference": ref, "title": study.get("title"),
                       "phase": study.get("phase"), "therapeutic_area": study.get("therapeutic_area"),
                       "rules": chosen}
    if new_name:
        new["name"] = new_name
    elif new["name"] in (P.default_profile()["name"], "SCOPE standard (customised)"):
        new["name"] = f"{study.get('protocol_number') or filename} study profile"
    new["version"] = P.bump_version(profile["version"])
    P.log_change(new, f"{len(chosen)} rules added from protocol {ref} ({filename})", who=who)
    return P.validate(new)


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def page_count_note(draft: dict) -> str:
    if draft["pages_read"] < draft["pages_total"]:
        return (f"The protocol has {draft['pages_total']} pages; SCOPE read the {draft['pages_read']} most relevant to "
                "monitoring (safety reporting, visits, eligibility, dosing, storage, deviations).")
    return f"SCOPE read all {draft['pages_total']} pages."

