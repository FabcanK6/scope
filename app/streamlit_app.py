"""SCOPE Streamlit front end (LLM engine).

    export GEMINI_API_KEY=...        # or add it to .streamlit/secrets.toml / the app's Secrets
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scope import profile as P  # noqa: E402
from scope import demos as DM  # noqa: E402
from scope import learning as L  # noqa: E402
from scope import ownmodel as OM  # noqa: E402
from scope import reports as RP  # noqa: E402
from scope import tracker as TR  # noqa: E402
from scope import protocol as PR  # noqa: E402
from scope import providers as PV  # noqa: E402
from scope.data.generate import read_jsonl  # noqa: E402
from scope.data.handwritten import load_handwritten, load_practice, load_realistic, load_stress  # noqa: E402
from scope.engine import ENGINE_REV, SYSTEM, LLMParser, Unreadable  # noqa: E402
from scope.llm import LLMError, draft_followup, get_api_key, verify_quote  # noqa: E402
from scope.record import audit_summary, to_row  # noqa: E402
from scope.schema import ISSUE_BY_CODE, ISSUE_GROUPS  # noqa: E402

CORPUS = ROOT / "data" / "corpus.jsonl"
RISK_COLORS = {"low": "#2E7D32", "medium": "#ED6C02", "high": "#C62828"}
SEVERITY_COLORS = {"critical": "#C62828", "major": "#ED6C02", "minor": "#B8860B"}
STATUS_COLORS = {"resolved_on_site": "#1565C0", "no_issue": "#2E7D32"}
MAX_NEW_CALLS = 25  # new LLM requests per browser session (cached answers are free)
PORTFOLIO_MAX = 10
# a new prompt or engine revision never reuses old cached answers
ENGINE_VERSION = hashlib.sha256((ENGINE_REV + SYSTEM).encode()).hexdigest()[:8]

st.set_page_config(page_title="SCOPE - Site Visit Note Intelligence", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# LLM access and caching
# ---------------------------------------------------------------------------
def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets configured
        return None


# ---------------------------------------------------------------------------
# Shared learning (scope/learning.py): corrections people share, approved by a curator, used for everyone
# ---------------------------------------------------------------------------
def _setting(name: str) -> str:
    import os

    return str(_secret(name) or os.environ.get(name) or "").strip()


@st.cache_resource(show_spinner=False)
def learning_store():
    """The shared library, if this copy of SCOPE has one (HF_TOKEN + SCOPE_LEARNING_REPO, or a local folder)."""
    try:
        if _setting("SCOPE_LEARNING_REPO") and _setting("HF_TOKEN"):
            return L.HFStore(_setting("SCOPE_LEARNING_REPO"), _setting("HF_TOKEN"))
        if _setting("SCOPE_LEARNING_DIR"):
            return L.LocalStore(_setting("SCOPE_LEARNING_DIR"))
    except L.LearningError:
        return None
    return None


@st.cache_data(ttl=120, show_spinner=False)
def _load_library() -> dict:
    try:
        return learning_store().load()
    except L.LearningError as e:  # remembered for 2 minutes too, so a outage doesn't slow every reading
        return {"_error": str(e)}


def library() -> dict | None:
    """Approved, pending and rejected cases (refreshed every 2 minutes); None when shared learning is off or down."""
    if learning_store() is None:
        return None
    lib = _load_library()
    if "_error" in lib:
        st.session_state["learning_error"] = lib["_error"]
        return None
    return lib


@st.cache_resource(show_spinner=False, max_entries=2)
def _own_model(digest: str, bar: float) -> OM.OwnModel:
    return OM.train(library(), bar=bar)


def own_model() -> OM.OwnModel | None:
    """SCOPE's own model, retrained by itself whenever the approved cases change (a few seconds)."""
    lib = library()
    try:
        bar = float(_setting("SCOPE_OWN_MODEL_BAR") or OM.ACTIVATION_BAR)
    except ValueError:
        bar = OM.ACTIVATION_BAR
    try:
        return _own_model(L.digest(lib) if lib else "-", bar)
    except Exception:  # never let the side model stop a reading
        return None


@st.cache_data(show_spinner=False)
def practice_notes() -> list[dict]:
    return [{k: r[k] for k in ("id", "text", "risk", "label", "tests") if k in r} for r in load_practice()]


VOTER_KEY = "scope.voter.v1"


def voter_id() -> str:
    """An anonymous id for this browser (a random value kept in its local storage), stored only as a hash."""
    if st.session_state.get("voter"):
        return st.session_state["voter"]
    raw = None
    if _browser_js is not None:
        raw = _browser_js(js_expressions=(
            f"localStorage.getItem('{VOTER_KEY}') || (localStorage.setItem('{VOTER_KEY}', "
            f"(self.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()), "
            f"localStorage.getItem('{VOTER_KEY}'))"), key="scope_voter")
    if raw:
        st.session_state["voter"] = L.voter_hash(str(raw))
        return st.session_state["voter"]
    import uuid

    return st.session_state.setdefault("voter_tmp", L.voter_hash(uuid.uuid4().hex))  # until the browser answers


def share_case(case: dict) -> bool:
    try:
        learning_store().add(case)
    except L.LearningError as e:
        st.error(f"Not shared: {e}")
        return False
    _load_library.clear()
    return True


SHARED = "App's free engine (Gemini)"
ENGINE_CHOICES = [SHARED, *PV.PROVIDERS]
KEY_HELP = {
    "Google Gemini": "Free key from aistudio.google.com.",
    "OpenAI": "Key from platform.openai.com (API keys). Usage is billed to your OpenAI account.",
    "Anthropic Claude": "Key from console.anthropic.com (API keys). Usage is billed to your Anthropic account.",
    "Other (OpenAI-compatible)": "Any service with an OpenAI-style API: Azure OpenAI, Mistral, Groq, OpenRouter, or a "
                                 "local Ollama or vLLM server (leave the key empty if it needs none).",
}


def engine_settings() -> dict:
    """Which LLM reads the notes: the app's shared Gemini key, or the visitor's own key for any provider."""
    choice = st.session_state.get("byo_provider", SHARED)
    if choice == SHARED:
        key = get_api_key(st.secrets if _secret("GEMINI_API_KEY") else None)
        return {"shared": True, "provider": "Google Gemini", "key": key, "model": _secret("GEMINI_MODEL"),
                "base": None}
    return {"shared": False, "provider": choice, "key": (st.session_state.get("byo_key") or "").strip(),
            "model": st.session_state.get("byo_model") or None, "base": st.session_state.get("byo_base") or None}


def api_key() -> str | None:
    """A fingerprint of the engine in use (provider, key, model, base URL) for the answer cache."""
    e = engine_settings()
    if not e["key"] and not (e["provider"].startswith("Other") and e["base"]):
        return None
    return f"{e['provider']}|{e['key']}|{e['model']}|{e['base']}"


def current_profile() -> dict:
    """The study profile in use in this browser session (default: SCOPE standard, rubric v3.3)."""
    if "profile" not in st.session_state:
        st.session_state["profile"] = P.default_profile()
    return st.session_state["profile"]


def get_parser() -> LLMParser | None:
    sig = api_key()
    if not sig:
        return None
    parser = st.session_state.get("parser")
    if parser is None or st.session_state.get("parser_sig") != sig:
        e = engine_settings()
        try:
            client = PV.make_client(e["provider"], e["key"], model=e["model"], base_url=e["base"])
        except LLMError as err:
            st.sidebar.error(str(err))
            return None
        parser = LLMParser(client)
        st.session_state["parser"], st.session_state["parser_sig"] = parser, sig
    parser.profile = current_profile()
    parser.library = library()
    return parser


@st.cache_resource
def shared_cache() -> dict:
    """Answers shared by all visitors for 24 hours, so repeated example notes cost no quota."""
    return {}


def _cache_key(kind: str, text: str, profile: dict | None = None) -> str:
    fp = hashlib.sha256((api_key() or "").encode()).hexdigest()[:12]
    prof = P.fingerprint(profile or current_profile())
    lib = library()
    learned = L.digest(lib) if lib else "-"  # what SCOPE has learned changes the answer
    return f"{kind}:{ENGINE_VERSION}:{prof}:{learned}:{fp}:{hashlib.sha256(text.encode()).hexdigest()}"


def cached_llm(kind: str, text: str, fn, profile: dict | None = None):
    cache, key = shared_cache(), _cache_key(kind, text, profile)
    hit = cache.get(key)
    if hit and time.time() - hit[0] < 24 * 3600:
        return hit[1]
    used = st.session_state.get("llm_calls", 0)
    if used >= MAX_NEW_CALLS and engine_settings()["shared"]:  # own keys are not capped
        raise LLMError(f"This session has used its {MAX_NEW_CALLS} new LLM requests. Reload the page later.")
    value = fn()  # failed requests (busy, quota) don't count against the session
    st.session_state["llm_calls"] = used + 1
    cache[key] = (time.time(), value)
    return value


@st.cache_resource(show_spinner=False)
def get_index():
    from scope.search import NoteIndex

    return NoteIndex(read_jsonl(CORPUS), backend="tfidf")


GENERAL_STUDY = "General: no protocol (SCOPE standard)"
OWN_STUDY = "New study: upload its protocol"
MY_PREFIX = "★ "
DEMO_BY_LABEL = {s["label"]: s for s in DM.index()}

try:  # keeps "My studies" in the user's own browser between visits (no accounts, nothing stored on the server)
    from streamlit_js_eval import streamlit_js_eval as _browser_js
except Exception:  # pragma: no cover - the app still works; studies then last for the session only
    _browser_js = None
BROWSER_KEY = "scope.studies.v1"


def my_studies() -> dict[str, dict]:
    if "my_studies" not in st.session_state:
        st.session_state["my_studies"] = {}
    return st.session_state["my_studies"]


def study_choices() -> list[str]:
    return [GENERAL_STUDY, *(MY_PREFIX + n for n in my_studies()), *DEMO_BY_LABEL, OWN_STUDY]


def remember_study(prof: dict, replaces: str | None = None) -> None:
    """Keep a study in "My studies" (saved in this browser) and select it on the next run."""
    if replaces and replaces != prof["name"]:
        my_studies().pop(replaces, None)
    my_studies()[prof["name"]] = prof
    st.session_state["profile"] = prof
    st.session_state["pending_study"] = MY_PREFIX + prof["name"]


TRACKER_KEY = "scope.tracker.v1"


def tracker() -> dict:
    """Site history and open actions for every study, kept in this browser (scope/tracker.py)."""
    return st.session_state.setdefault("tracker", {})


def tracker_load() -> None:
    """Read the site history from the browser once (merged with anything recorded before it answered)."""
    if _browser_js is None or st.session_state.get("tracker_loaded"):
        return
    raw = _browser_js(js_expressions=f"localStorage.getItem('{TRACKER_KEY}') || '{{}}'", key="scope_tr_read")
    if raw is None:
        return
    try:
        saved = json.loads(raw)
        saved = saved if isinstance(saved, dict) else {}
    except ValueError:
        saved = {}
    for study, sites in saved.items():
        for site, visits in (sites or {}).items() if isinstance(sites, dict) else []:
            for v in visits if isinstance(visits, list) else []:
                if isinstance(v, dict) and v.get("key"):
                    TR.add_visit(tracker(), study, site, v)
    st.session_state["tracker_loaded"] = True
    st.session_state["tracker_saved"] = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def tracker_save() -> None:
    """Write the site history back to the browser whenever it changed (called at the end of each run)."""
    if _browser_js is None or not st.session_state.get("tracker_loaded"):
        return
    payload = json.dumps(tracker(), sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if st.session_state.get("tracker_saved") != digest:
        _browser_js(js_expressions=f"localStorage.setItem('{TRACKER_KEY}', {json.dumps(payload)}); 'saved'",
                    key=f"scope_tr_write_{digest[:12]}")
        st.session_state["tracker_saved"] = digest


def site_history_panel(rec: dict, note: str) -> None:
    """Record this visit in the site's history and show what is still open from earlier visits."""
    site = TR.site_key(rec)
    if not site:
        st.caption("No site number in this note, so it was not added to a site history.")
        return
    study = current_profile()["name"]
    entry = TR.visit_entry(rec, note)
    TR.add_visit(tracker(), study, site, entry)
    for h in TR.repeat_hints(tracker(), study, site, rec, note):
        when = h["last_date"] or "date not given"
        st.info(f"**{h['display']}** was also active at this site's previous visit ({when}, {h['last_severity']}). "
                "If it is the same problem, the rubric treats it as a repeat finding (one level higher); the note "
                "does not say so, so SCOPE has not raised it.")
    still_open = TR.open_actions(tracker(), study, site, entry["key"])
    label = f"Still open from earlier visits to site {site} ({len(still_open)})"
    with st.expander(label, expanded=bool(still_open)):
        if not still_open:
            st.caption("Nothing open from earlier visits. This visit is saved to the site history in this browser.")
        for a in still_open:
            due = f" · due {a['due']}" if a.get("due") else ""
            owner = f" · {a['owner']}" if a.get("owner") else ""
            if st.checkbox(f"{a['action']}{owner}{due} (visit {a['visit_date'] or '?'})", key=f"done_{a['id']}"):
                TR.set_done(tracker(), study, site, a["id"], True)
                st.rerun()
        if still_open:
            st.caption("Tick an item once this note (or anything else) shows it is done. Saved in this browser.")


def sites_tab() -> None:
    """Every site in the current study: visits, risk over time and open action items."""
    study = current_profile()["name"]
    st.markdown(f"**Site history for {study}.** Each visit you read for this study is added to its site, so the next "
                "visit starts with what is still open. Kept in this browser only.")
    rows = TR.site_summary(tracker(), study)
    if not rows:
        st.info("No visits yet for this study. Read a note that names its site, and it appears here.")
        return
    st.dataframe(pd.DataFrame([{"Site": r["site"], "Visits": r["visits"], "Last visit": r["last_visit"],
                                "Last risk": r["last_risk"].upper(), "Open actions": r["open_actions"]}
                               for r in rows]), hide_index=True, width="stretch")
    site = st.selectbox("Site", [r["site"] for r in rows], key="sites_pick")
    visits = tracker()[study][site]
    st.subheader(f"Open action items, site {site}")
    actions = [{**a, "visit_date": v["date"]} for v in visits for a in v["actions"]]
    if actions:
        df = pd.DataFrame([{"Done": bool(a.get("done")), "Action": a["action"], "Owner": a["owner"], "Due": a["due"],
                            "From visit": a["visit_date"], "id": a["id"]} for a in actions])
        edited = st.data_editor(df, key=f"actions_{site}", hide_index=True, width="stretch",
                                disabled=["Action", "Owner", "Due", "From visit", "id"],
                                column_config={"id": None})
        changed = False
        for _, r in edited.iterrows():
            before = next(a for a in actions if a["id"] == r["id"])
            if bool(r["Done"]) != bool(before.get("done")):
                TR.set_done(tracker(), study, site, r["id"], bool(r["Done"]))
                changed = True
        if changed:
            st.rerun()
    else:
        st.caption("No action items recorded for this site.")
    st.subheader("Visits")
    st.dataframe(pd.DataFrame([{"Date": v["date"], "Type": v["visit_type"], "Risk": v["risk"].upper(),
                                "Active findings": ", ".join(f"{f['display']} ({f['severity']})"
                                                             for f in v["findings"]) or "none",
                                "Actions": len(v["actions"])} for v in reversed(visits)]),
                 hide_index=True, width="stretch")
    if st.button(f"Forget site {site}'s history in this browser"):
        tracker()[study].pop(site, None)
        st.rerun()


def browser_sync() -> None:
    """Load saved studies from the browser once, then save them whenever they change."""
    if _browser_js is None:
        return
    if not st.session_state.get("studies_loaded"):
        raw = _browser_js(js_expressions=f"localStorage.getItem('{BROWSER_KEY}') || '{{}}'", key="scope_ls_read")
        if raw is None:  # the browser has not answered yet
            return
        try:
            saved = json.loads(raw).get("studies", {})
        except (ValueError, AttributeError):
            saved = {}
        loaded = {}
        for name, prof in saved.items():
            try:
                loaded[name] = P.validate(prof)
            except P.ProfileError:
                continue
        st.session_state["my_studies"] = {**loaded, **my_studies()}
        st.session_state["studies_loaded"] = True
        st.session_state["studies_saved"] = hashlib.sha256(json.dumps({"studies": loaded}).encode()).hexdigest()
        if loaded:
            st.rerun()
    payload = json.dumps({"studies": my_studies()})
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if st.session_state.get("studies_saved") != digest:
        _browser_js(js_expressions=f"localStorage.setItem('{BROWSER_KEY}', {json.dumps(payload)}); 'saved'",
                    key=f"scope_ls_write_{digest[:12]}")
        st.session_state["studies_saved"] = digest


def chosen_demo() -> dict | None:
    return DEMO_BY_LABEL.get(st.session_state.get("study_choice", GENERAL_STUDY))


def note_choices() -> dict[str, dict]:
    """Example notes for the chosen study: its demo notes, or the general examples."""
    demo = chosen_demo()
    if demo:
        return {n["title"]: n for n in DM.notes(demo["id"])}
    return {label: {"text": text} for label, text in examples().items()}


def select_study() -> None:
    choice = st.session_state.get("study_choice", GENERAL_STUDY)
    demo = DEMO_BY_LABEL.get(choice)
    if demo:
        st.session_state["profile"] = DM.profile(demo["id"])
    elif choice.startswith(MY_PREFIX) and choice[len(MY_PREFIX):] in my_studies():
        st.session_state["profile"] = my_studies()[choice[len(MY_PREFIX):]]
    elif choice == GENERAL_STUDY:
        st.session_state["profile"] = P.default_profile()
    st.session_state["example_pick"] = None
    st.session_state.pop("expected", None)


def analyze_under(text: str, prof: dict | None = None) -> dict:
    """Score a note for the accuracy check: under a given study profile (demo notes), and never with a ruling that
    was made on this same note, so SCOPE is not shown the answer."""
    saved = parser.profile
    parser.profile = prof or saved
    parser.holdout = True
    try:
        return cached_llm("check", text, lambda: parser.analyze(text), profile=parser.profile)
    finally:
        parser.profile, parser.holdout = saved, False


@st.cache_data
def examples() -> dict[str, str]:
    notes = {r["id"]: r["text"] for r in load_handwritten() + load_realistic()}
    picks = {"Formal IMV report (clean)": "real-01", "For-cause visit (late SAE)": "real-05",
             "SIV with a pending item": "real-04", "IMV with a deviation": "real-03",
             "Quick field note": "hw-01", "Missed SAE": "hw-04", "E-mail style": "hw-18", "Bullet notes": "hw-16"}
    return {label: notes[i] for label, i in picks.items()}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_llm_error(e: LLMError) -> None:
    st.error(str(e))
    if getattr(e, "engine_log", None):
        with st.expander("What SCOPE tried"):
            st.markdown("\n".join(f"- {line}" for line in e.engine_log))
    if isinstance(e, Unreadable) and e.raw:
        with st.expander("What the AI model returned (not used)"):
            st.code(e.raw[:6000])


def risk_badge(level: str) -> str:
    return (f"<span style='background:{RISK_COLORS[level]};color:white;padding:5px 14px;border-radius:14px;"
            f"font-weight:700;font-size:1.05rem'>{level.upper()} RISK</span>")


def risk_reason(rec: dict) -> str:
    sev = [i["severity"] for i in rec["issues"] if i.get("severity")]
    if not sev:
        return "No active findings."
    counts = {s: sev.count(s) for s in ("critical", "major", "minor") if s in sev}
    parts = [f"{n} {s}" for s, n in counts.items()]
    return "Because of " + ", ".join(parts) + " finding" + ("s" if len(sev) > 1 else "") + " (severity rubric)."


def highlight(text: str, rec: dict) -> str:
    regions = []
    for f in rec.get("findings", []):
        if f.get("verified") and f.get("char_start", -1) >= 0:
            sev = f.get("final_severity", f["severity"])
            color = SEVERITY_COLORS[sev] if f["status"] == "active" else STATUS_COLORS[f["status"]]
            label = f"{f['display']} · {sev if f['status'] == 'active' else f['status'].replace('_', ' ')}"
            regions.append((f["char_start"], f["char_end"], color, label))
    for sp in rec["spans"]:
        if sp["label"] in ("ACTION", "OWNER", "DUE", "VISIT_DATE", "VISIT_TYPE", "SITE"):
            regions.append((sp["char_start"], sp["char_end"], "#6A1B9A", sp["label"].replace("_", " ").lower()))
    out, pos = [], 0
    for s, e, color, label in sorted(regions):
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        out.append(f"<span title='{html.escape(label)}' style='background:{color}22;border-bottom:2px solid "
                   f"{color};padding:1px 2px;border-radius:3px'>{html.escape(text[s:e])}"
                   f"<sub style='color:{color};font-size:0.62em;margin-left:3px'>{html.escape(label)}</sub></span>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "<div style='line-height:2.1;font-size:0.95rem'>" + "".join(out).replace("\n", "<br>") + "</div>"


def findings_table(rec: dict, status: str) -> pd.DataFrame:
    rows = [{"Issue": f["display"], "Severity": f.get("final_severity", f["severity"]),
             "Severity set by": "; ".join(([f["rule_applied"]] if f.get("rule_applied") else [])
                                         + [f"raised: {x}" for x in f.get("escalated_by") or []]),
             "Evidence (quoted from the note)": f["evidence"], "Why": f.get("explanation", "")}
            for f in rec.get("findings", []) if f["status"] == status and f.get("verified")]
    order = {"critical": 0, "major": 1, "minor": 2}
    rows.sort(key=lambda r: order.get(r["Severity"], 3))
    if status != "active":
        for r in rows:
            r.pop("Severity")
            r.pop("Severity set by")
    elif not any(r["Severity set by"] for r in rows):
        for r in rows:
            r.pop("Severity set by")
    return pd.DataFrame(rows)


def feedback(note: str, rec: dict) -> None:
    """Was SCOPE right? A one-click confirmation (shared, when shared learning is on) or a correction."""
    store = learning_store()
    if store is not None:
        key = f"confirmation:{L.note_hash(note)}:{rec['risk']['level']}"
        done = key in st.session_state.setdefault("shared", set())
        c1, c2 = st.columns([1, 2.4])
        if c1.button("SCOPE got this right", key=f"ok_{key}", disabled=done):
            if share_case(L.make_case("confirmation", note, rec, current_profile(), voter=voter_id())):
                st.session_state["shared"].add(key)
                done = True
        c2.caption("Shared. Once two more people agree, SCOPE learns from it. Thank you!" if done else
                   "One click teaches SCOPE. It shares this note and SCOPE's reading with other SCOPE users for "
                   "review: fictional or de-identified notes only.")
    correction_form(note, rec, store)


def correction_form(note: str, rec: dict, store=None) -> None:
    """Let a user correct SCOPE. Corrections are proposed; the study lead approves them in the Study setup tab, and
    a shared correction is reviewed for the shared library."""
    prof = current_profile()
    topics = P.enabled_topics(prof)
    found = [f for f in rec.get("findings", []) if f.get("verified")]
    with st.expander("Disagree with SCOPE? Correct it"):
        st.caption("Your correction is saved to this study as a proposal. Once the study lead approves it, SCOPE "
                   "uses it whenever it reads a similar note." + (" Share it too, and once two more people agree, "
                                                                  "it helps every SCOPE user." if store else ""))
        with st.form(f"correct_{hashlib.sha256(note.encode()).hexdigest()[:8]}", clear_on_submit=True):
            options = [f"{f['display']} ({f['status'].replace('_', ' ')}"
                       f"{', ' + f.get('final_severity', f['severity']) if f['status'] == 'active' else ''})"
                       for f in found] + ["Something SCOPE missed"]
            pick = st.selectbox("Which finding?", options)
            idx = options.index(pick)
            chosen = found[idx] if idx < len(found) else None
            topic_names = [t["display"] for t in topics]
            default_topic = topic_names.index(chosen["display"]) if chosen and chosen["display"] in topic_names else 0
            topic = st.selectbox("Topic", topic_names, index=default_topic)
            status = st.radio("It should be", ["active problem", "fixed during the visit", "not a problem"],
                              horizontal=True)
            severity = st.select_slider("Severity (if active)", ["minor", "major", "critical"],
                                        value=chosen.get("severity", "minor") if chosen else "minor")
            quote = st.text_input("Words from the note that show it", value=chosen["evidence"] if chosen else "")
            reason = st.text_input("Why (one line)")
            cra_risk = st.selectbox("What should this visit's risk be? (optional)", ["", "low", "medium", "high"])
            who = st.text_input("Your name or initials (optional)")
            share = store is not None and st.checkbox(
                "Share it with other SCOPE users so SCOPE learns for everyone (fictional or de-identified notes only)")
            if st.form_submit_button("Save correction"):
                if not quote.strip() or not verify_quote(quote, note):
                    st.error("The quote must be words copied from the note.")
                    return
                code = next(t["code"] for t in topics if t["display"] == topic)
                stat = {"active problem": "active", "fixed during the visit": "resolved_on_site",
                        "not a problem": "no_issue"}[status]
                scope_said = (f"{chosen['status']}, {chosen.get('final_severity', chosen['severity'])}"
                              if chosen else "missed")
                P.add_correction(prof, note=note, issue=code, status=stat, severity=severity, quote=quote,
                                 reason=reason, scope_said=scope_said, cra_risk=cra_risk or None, who=who)
                msg = "Correction saved for this study as a proposal (approve it in the Study setup tab)."
                if share:
                    case = L.make_case("correction", note, rec, prof, by=who, voter=voter_id(), correction={
                        "issue": code, "display": topic, "status": stat,
                        "severity": severity if stat == "active" else None, "quote": quote.strip(),
                        "reason": reason.strip(), "risk": cra_risk or None})
                    if share_case(case):
                        msg += (" Shared: once two more people agree in **Help SCOPE learn**, SCOPE uses it for "
                                "everyone.")
                st.success(msg)


def own_model_status() -> None:
    st.subheader("SCOPE's own model")
    om = own_model()
    if om is None:
        st.caption("SCOPE's own model could not be trained right now.")
        return
    r = om.report
    if r["accuracy"] is None:
        st.caption("Not enough expert-labelled notes yet to measure it.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Agrees with people", f"{r['accuracy']:.0%}", help="Risk level, on notes labelled by experts or "
              "agreed by the community that it was not trained on (5-fold check, repeated 3 times).")
    c2.metric("High-risk visits caught", f"{r['high_recall']:.0%}" if r["high_recall"] is not None else "-")
    c3.metric("Trained on", f"{r['expert_notes'] + r['community_notes'] + r['shared_cases']} notes",
              help=f"{r['expert_notes']} expert-labelled notes, {r['community_notes']} practice notes the "
                   f"community agreed on and {r['shared_cases']} approved shared cases")
    if om.active:
        st.success("Switched on: it gives a second opinion on every visit's risk level, and a rough estimate when the "
                   "AI model is unavailable.")
    else:
        st.info(f"Not switched on yet: it switches itself on at {r['bar']:.0%} agreement (and "
                f"{r['high_recall_bar']:.0%} of high-risk visits caught). It retrains by itself every time a shared "
                "case is approved or a practice note is agreed, so every rating moves it closer.")


def practice_panel(lib: dict, store) -> None:
    """Rate fictional practice notes: the community's labels, counted once enough people agree."""
    st.subheader("Rate a practice visit")
    ratings = sum(v["item"].startswith("note:") for v in lib.get("votes", []))
    raters = len({v["voter"] for v in lib.get("votes", [])})
    st.caption("Fictional notes. Read the note and pick the risk you would give the visit. Ratings are anonymous; a "
               f"label counts once {L.AGREE_MIN} people agree, then SCOPE learns from it."
               + (f" So far: {ratings} ratings from {raters} people." if ratings else " Be one of the first to rate."))
    me = voter_id()
    notes = practice_notes()
    mine = {v["item"] for v in lib.get("votes", []) if v["voter"] == me}
    skipped = st.session_state.setdefault("practice_skipped", set())
    last = st.session_state.get("practice_last")
    if last:
        values = [v["value"] for v in L.votes_for(lib, f"note:{last['id']}")]
        tally = " · ".join(f"{r} {values.count(r)}" for r in L.RISKS)
        agreed = L.note_consensus(values)
        st.success(f"Thanks! You said **{last['value'].upper()}**. Ratings so far: {tally}"
                   + (f" (agreed: **{agreed.upper()}**)." if agreed else ".")
                   + f" SCOPE's suggested label was **{last['proposed'].upper()}**: {last['label']}")
    todo = [n for n in notes if f"note:{n['id']}" not in mine and n["id"] not in skipped]
    done = sum(f"note:{n['id']}" in mine for n in notes)
    st.caption(f"You have rated {done} of {len(notes)} practice notes.")
    if not todo:
        st.info("You have rated every practice note. Thank you! New ones are added over time.")
        return
    note = todo[0]
    with st.container(border=True):
        st.text(note["text"])
    with st.form(f"rate_{note['id']}", clear_on_submit=True):
        risk = st.radio("What risk would you give this visit?", list(L.RISKS), index=None, horizontal=True)
        why = st.text_input("Main reason (optional)")
        c1, c2, _ = st.columns([1, 1, 3])
        submit = c1.form_submit_button("Submit rating", type="primary")
        skip = c2.form_submit_button("Skip")
    if skip:
        skipped.add(note["id"])
        st.session_state.pop("practice_last", None)
        st.rerun()
    if submit:
        if not risk:
            st.warning("Pick low, medium or high first.")
            return
        try:
            store.add_vote(L.make_vote(f"note:{note['id']}", me, risk, why))
        except L.LearningError as e:
            st.error(f"Not saved: {e}")
            return
        _load_library.clear()
        st.session_state["practice_last"] = {"id": note["id"], "value": risk, "proposed": note["risk"],
                                             "label": note.get("label", "")}
        st.rerun()


def community_review(lib: dict, store) -> None:
    """Shared corrections and confirmations, reviewed by other SCOPE users: three agreeing makes them count."""
    st.subheader("Review what others shared")
    if st.session_state.get("review_note"):
        st.success(st.session_state.pop("review_note"))
    me = voter_id()
    mine = {v["item"] for v in lib.get("votes", []) if v["voter"] == me}
    queue = [c for c in lib["pending"] if c.get("voter") != me and f"case:{c['id']}" not in mine]
    if not queue:
        st.caption("Nothing waiting for you to review right now.")
        return
    st.caption(f"Does SCOPE have this right? A shared case goes live once {L.AGREE_MIN} people agree (the person who "
               "shared it counts as one), and is dropped if as many disagree.")
    topics = P.topic_map(current_profile())
    for c in queue[:5]:
        with st.container(border=True):
            said = c.get("scope_said") or {}
            active = [f"{topics.get(f['issue'], {}).get('display', f['issue'])} ({f['severity']})"
                      for f in said.get("findings", []) if f.get("status") == "active"]
            st.caption(f"SCOPE read this visit as {str(said.get('risk') or '-').upper()} risk"
                       + (f": {', '.join(active)}" if active else ", no active problems"))
            with st.expander("The note"):
                st.text(c["note"])
            if c["kind"] == "correction":
                x = c["correction"]
                st.markdown("**Someone says:** " + L.ruling_line(c, topics)[2:]
                            + (f" Visit risk should be **{x['risk'].upper()}**." if x.get("risk") else ""))
                question = "Do you agree with this correction?"
            else:
                st.markdown("**Someone says SCOPE's reading is right.**")
                question = "Do you agree SCOPE read it right?"
            agree, disagree = L.case_tally(c, lib)
            st.caption(f"{question} So far {agree} agree, {disagree} disagree.")
            b1, b2, _ = st.columns([1, 1, 3])
            for value, btn in (("agree", b1.button("Agree", key=f"ag_{c['id']}")),
                               ("disagree", b2.button("Disagree", key=f"dis_{c['id']}"))):
                if btn:
                    try:
                        store.add_vote(L.make_vote(f"case:{c['id']}", me, value))
                        fresh = store.load()
                        decided = L.apply_consensus(store, fresh)
                    except L.LearningError as e:
                        st.error(f"Not saved: {e}")
                        break
                    _load_library.clear()
                    if any(d["id"] == c["id"] and d["status"] == "approved" for d in decided):
                        st.session_state["review_note"] = "Agreed by the community: SCOPE now uses it."
                    else:
                        st.session_state["review_note"] = "Thanks, your vote is in."
                    st.rerun()


def learned_tab() -> None:
    """Help SCOPE learn: rate practice notes, review what others shared, and see what SCOPE has learned."""
    st.markdown("**SCOPE learns from the people who use it.** Rate a practice visit, or agree or disagree with what "
                "others shared. When enough people agree, SCOPE uses it straight away, for everyone. Every ruling "
                "keeps who agreed, when and why.")
    store = learning_store()
    if store is None:
        st.info("Shared learning is not switched on for this copy of SCOPE. Corrections still improve each study "
                "(see Study setup).")
        return
    lib = library()
    if lib is None:
        st.warning("The shared library can't be reached right now, so SCOPE is reading notes without it. "
                   + st.session_state.get("learning_error", ""))
        return
    practice_panel(lib, store)
    st.divider()
    community_review(lib, store)
    st.divider()
    st.subheader("What SCOPE has learned")
    approved = lib["approved"]
    rulings_ = [c for c in approved if c["kind"] == "correction"]
    agreed = L.community_rows(lib, practice_notes())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rulings in use", len(rulings_))
    m2.metric("Confirmed readings", len(approved) - len(rulings_))
    m3.metric("Practice notes agreed", len(agreed))
    m4.metric("Waiting for review", len(lib["pending"]))
    if rulings_:
        st.dataframe(pd.DataFrame([{
            "Since": c.get("decided") or "", "When a note says": c["correction"]["quote"],
            "Ruling": L.ruling_line(c, P.topic_map(current_profile())).split(": ", 1)[-1],
            "Applies to": "All studies" if c.get("applies_to") == L.ALL_STUDIES else c.get("applies_to"),
            "Agreed by": c.get("curator_note") or "curator"}
            for c in reversed(rulings_)]), hide_index=True, width="stretch")
    else:
        st.caption("No rulings yet. Correct a reading and tick **Share it** to teach SCOPE the first one.")
    own_model_status()
    st.divider()
    st.markdown("**Know someone who reviews visit notes?** Every person who rates a few practice visits makes SCOPE "
                "better for the next one. Share the app:")
    st.code("https://scope-fabcank6.streamlit.app", language=None)

    with st.expander("Curator tools"):
        expected = _setting("SCOPE_CURATOR_KEY")
        if not expected:
            st.caption("Add SCOPE_CURATOR_KEY to the app's secrets to review shared cases here.")
            return
        given = st.text_input("Curator key", type="password", key="curator_key")
        if not given:
            return
        if not hmac.compare_digest(given.encode(), expected.encode()):
            st.error("That key is not right.")
            return
        rows = L.training_rows(lib)
        st.download_button("Download training data (.jsonl)", "\n".join(json.dumps(r) for r in rows),
                           file_name="scope_training.jsonl", disabled=not rows)
        st.caption("Curators can approve or reject a shared case without waiting for the community, and retire any "
                   "ruling.")
        if not lib["pending"]:
            st.success("Nothing is waiting for review.")
        topics = P.topic_map(current_profile())
        for c in lib["pending"][:20]:
            with st.container(border=True):
                study = (c.get("study") or {}).get("name") or "SCOPE standard"
                who = f" by {c['by']}" if c.get("by") else ""
                st.markdown(f"**{c['kind'].capitalize()}**{who} · {c['created'][:10]} · study: {study}")
                said = c.get("scope_said") or {}
                active = [f"{f['issue']} ({f['severity']})" for f in said.get("findings", [])
                          if f.get("status") == "active"]
                st.caption(f"SCOPE said: {str(said.get('risk') or '-').upper()} risk"
                           + (f"; active: {', '.join(active)}" if active else "; no active findings"))
                if c["kind"] == "correction":
                    x = c["correction"]
                    st.markdown("Reviewer says " + L.ruling_line(c, topics)[2:]
                                + (f" Visit risk should be **{x['risk']}**." if x.get("risk") else ""))
                with st.expander("Note"):
                    st.text(c["note"])
                scope_choice = L.ALL_STUDIES
                if c["kind"] == "correction" and study != "SCOPE standard":
                    pick = st.radio("Applies to", ["All studies", f"This study only ({study})"], horizontal=True,
                                    key=f"to_{c['id']}")
                    scope_choice = L.ALL_STUDIES if pick == "All studies" else study
                why = st.text_input("Note for the record (optional)", key=f"why_{c['id']}")
                b1, b2, _ = st.columns([1, 1, 3])
                for approve, btn in ((True, b1.button("Approve", key=f"yes_{c['id']}", type="primary")),
                                     (False, b2.button("Reject", key=f"no_{c['id']}"))):
                    if btn:
                        try:
                            store.decide(c, approve=approve, applies_to=scope_choice, note=why)
                        except L.LearningError as e:
                            st.error(str(e))
                            break
                        _load_library.clear()
                        st.rerun()
        if rulings_:
            st.markdown("**Rulings in use.** Retire one when it no longer holds (a rubric change, a wrong call): "
                        "SCOPE stops using it at once, and it stays in the record as retired.")
            for c in reversed(rulings_):
                r1, r2 = st.columns([5, 1])
                r1.caption(L.ruling_line(c, topics)[2:])
                if r2.button("Retire", key=f"retire_{c['id']}"):
                    try:
                        store.decide(c, approve=False, note="Retired by curator")
                    except L.LearningError as e:
                        st.error(str(e))
                    else:
                        _load_library.clear()
                        st.rerun()


def _profile_diff(old: dict, new: dict) -> str:
    """One line describing what changed between two profiles (for the change log)."""
    parts = []
    o, n = P.topic_map(old), P.topic_map(new)
    added = [n[c]["display"] for c in n if c not in o]
    removed = [o[c]["display"] for c in o if c not in n]
    toggled = [n[c]["display"] for c in n if c in o and n[c]["enabled"] != o[c]["enabled"]]
    edited = [n[c]["display"] for c in n if c in o and any(n[c][k] != o[c][k] for k in P.SEVERITIES + ["display"])]
    for label, items in (("added", added), ("removed", removed), ("switched on/off", toggled), ("edited", edited)):
        if items:
            parts.append(f"topics {label}: {', '.join(items)}")
    if old["study_rules"] != new["study_rules"]:
        parts.append(f"study rules now {len(new['study_rules'])}")
    if old["escalation"] != new["escalation"]:
        parts.append("escalation settings changed")
    if old.get("deadlines") != new.get("deadlines"):
        parts.append("reporting deadlines changed")
    if old["thresholds"] != new["thresholds"]:
        parts.append(f"risk thresholds medium {new['thresholds']['medium']}, high {new['thresholds']['high']}")
    oa = sum(bool(c.get("approved")) for c in old["corrections"])
    na = sum(bool(c.get("approved")) for c in new["corrections"])
    if (oa, len(old["corrections"])) != (na, len(new["corrections"])):
        parts.append(f"corrections: {na} approved of {len(new['corrections'])}")
    if old["name"] != new["name"]:
        parts.append(f"renamed to {new['name']}")
    return "; ".join(parts)


def sidebar_protocol() -> None:
    """Upload a protocol in the sidebar; the drafted rules are reviewed in the Study setup tab."""
    upload = st.file_uploader("Upload the protocol (PDF, Word or text)", type=["pdf", "docx", "txt"],
                              key="pr_file", help="SCOPE drafts the study's rules from it (SAE definitions and "
                              "deadlines, visit windows, eligibility, dosing, storage, deviations). You review them "
                              "before anything is used.")
    source = (upload.name, upload.getvalue()) if upload is not None else None
    go = source is not None and st.button("Read the protocol", type="primary", width="stretch")
    st.caption("Only public (e.g. ClinicalTrials.gov) or fictional protocols on the free engine.")
    if go and source:
        if not parser_for_sidebar():
            st.info("Set up the AI engine below first.")
            return
        try:
            pages = PR.read_document(*source)
            base = current_profile()
            key = f"{PR.fingerprint(source[1])}:{P.fingerprint(base)}"
            with st.spinner("Reading the protocol..."):
                draft = cached_llm("protocol", key, lambda: PR.draft_rules(parser_for_sidebar().client, pages, base))
            st.session_state["protocol_draft"] = {"file": source[0], "draft": draft}
        except LLMError as e:
            st.error(str(e))
    pd_state = st.session_state.get("protocol_draft")
    if pd_state:
        st.success(f"{len(pd_state['draft']['rules'])} rules drafted from {pd_state['file']}. Review them in the "
                   "**Study setup** tab.")


def parser_for_sidebar():
    return get_parser()


def protocol_review() -> None:
    """Review the rules drafted from a protocol and turn the accepted ones into a study profile."""
    pd_state = st.session_state.get("protocol_draft")
    if not pd_state:
        st.info("To use your own study, choose **New study: upload its protocol** in the sidebar (step 1). SCOPE "
                "drafts the study's rules from it, each with the exact quote and page, and you review them here before "
                "anything is used. Or pick a demo study to see a finished profile.")
        return
    st.subheader("Rules drafted from the protocol")
    presets = {"SCOPE standard": P.default_profile, "The current profile": current_profile}
    base_choice = st.selectbox("Add the accepted rules to", list(presets), key="pr_base")
    base = presets[base_choice]()
    draft = pd_state["draft"]
    study = draft["study"]
    st.markdown(f"**{study.get('protocol_number') or pd_state['file']}** "
                f"{study.get('version') or ''} · {study.get('title') or ''}  \n"
                f"Phase {study.get('phase') or '?'} · {study.get('therapeutic_area') or 'therapeutic area not stated'}")
    st.caption(PR.page_count_note(draft) + (f" {draft['dropped']} drafted rule(s) were dropped because their quote "
                                            "is not in the protocol." if draft["dropped"] else ""))
    if not draft["rules"]:
        st.info("No study-specific rules were found.")
        return
    names = {t["code"]: t["display"] for t in base["topics"]}
    df = pd.DataFrame([{"Accept": True, "Topic": names.get(r["topic"], "Whole study"), "Rule": r["rule"],
                        "If broken": r["severity"], "Quote from the protocol": r["quote"], "Page": r["page"]}
                       for r in draft["rules"]])
    edited = st.data_editor(df, key="pr_rules", hide_index=True, width="stretch",
                            disabled=["Topic", "Quote from the protocol", "Page"],
                            column_config={"Accept": st.column_config.CheckboxColumn(width="small"),
                                           "If broken": st.column_config.SelectboxColumn(
                                               options=PR.SEVERITY_OPTIONS)})
    st.caption("You can reword a rule or change what breaking it counts as before accepting. 'definition' means the "
               "rule changes what counts (for example, 'disease progression is not an SAE').")
    c1, c2 = st.columns([2, 1])
    default_name = f"{study.get('protocol_number') or pd_state['file']} study profile"
    name = c1.text_input("Name for the new profile", default_name, key="pr_name")
    who = c2.text_input("Your name or initials", key="pr_who")
    if st.button("Create the study profile from the accepted rules", type="primary"):
        rules = [{**r, "rule": str(row["Rule"]).strip() or r["rule"], "severity": row["If broken"]}
                 for r, (_, row) in zip(draft["rules"], edited.iterrows())]
        accepted = [i for i, (_, row) in enumerate(edited.iterrows()) if bool(row["Accept"])]
        if not accepted:
            st.error("Accept at least one rule.")
            return
        new = PR.apply_rules(base, {**draft, "rules": rules}, accepted, pd_state["file"], who=who.strip(),
                             new_name=name.strip() or None)
        remember_study(new)
        st.session_state.pop("protocol_draft", None)
        st.success(f"Created {new['name']} v{new['version']} with {len(accepted)} protocol rules. It is saved under "
                   "My studies (★) in this browser: pick it next time, no upload needed.")
        st.rerun()


def profile_editor() -> None:
    prof = current_profile()
    st.markdown(
        "A study profile holds the rules SCOPE scores this study by. It starts from **SCOPE standard** (a severity "
        "rubric written by an experienced clinical research professional) plus the rules accepted from the protocol. "
        "Adjust it below: what counts as minor, major or critical, study-specific topics and rules, escalation, and "
        "corrections for SCOPE to learn from.")
    st.info("**Save changes** keeps this study in this browser under ★ in the study list, ready for your next visit. "
            "To use it on another computer or share it, use **Download this profile** in the sidebar. Every result "
            "shows which profile and version scored it.")
    protocol_review()
    st.divider()
    st.subheader("This profile")
    if prof.get("protocol"):
        pr = prof["protocol"]
        st.caption(f"Built from protocol {pr.get('reference')} ({pr.get('file')}): {len(pr.get('rules', []))} rules, "
                   "cited in the study rules below.")
    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.text_input("Profile name", prof["name"], key="pf_name")
    c2.metric("Version", prof["version"])
    c3.metric("Fingerprint", P.fingerprint(prof))

    st.subheader("Topics and severities")
    st.caption("Edit the text that defines each severity, switch topics off that do not apply, or add a row for a "
               "study-specific topic. Leave a severity empty if it does not exist for that topic.")
    topics_df = pd.DataFrame([{"On": t["enabled"], "Topic": t["display"], "Group": t["group"], "Minor": t["minor"],
                               "Major": t["major"], "Critical": t["critical"], "Code": t["code"]}
                              for t in prof["topics"]])
    edited_topics = st.data_editor(
        topics_df, key="pf_topics", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"On": st.column_config.CheckboxColumn(width="small"),
                       "Code": st.column_config.TextColumn(disabled=True, help="Set automatically for new topics"),
                       "Group": st.column_config.SelectboxColumn(options=P.GROUPS + ["Study-specific"])})

    st.subheader("Study-specific rules")
    rules = st.text_area("One rule per line. These come from the protocol and override the rubric.",
                         "\n".join(prof["study_rules"]), key="pf_rules", height=110,
                         placeholder="A missed Week 4 PK sample is critical (protocol deviation).\n"
                                     "Visit windows are plus or minus 7 days; up to 7 days late is not a deviation.")

    st.subheader("Escalation and risk thresholds")
    e1, e2, e3, e4, e5 = st.columns(5)
    esc, th = prof["escalation"], prof["thresholds"]
    n_subj = e1.number_input("Subjects that make a problem widespread", 2, 50, esc["subjects_threshold"],
                             key="pf_subj")
    cap = e2.selectbox("Widespread raises it up to", P.SEVERITIES, P.SEVERITIES.index(esc["subjects_max"]),
                       key="pf_cap")
    rep_on = e3.checkbox("Repeat findings go up one level", esc["repeat"], key="pf_repeat")
    med = e4.number_input("Points for medium risk", 1, 30, th["medium"], key="pf_med")
    high = e5.number_input("Points for high risk", 2, 60, th["high"], key="pf_high")
    st.caption("Points: minor 1, major 3, critical 6, worst finding per topic.")

    st.subheader("Reporting deadlines")
    st.caption("SCOPE counts these itself from the dates in each note (business days are Monday to Friday). A "
               "protocol's own deadlines replace the defaults when you accept them.")
    code_to_name = {t["code"]: t["display"] for t in prof["topics"]}
    name_to_code = {v: k for k, v in code_to_name.items()}
    dl_df = pd.DataFrame([{"Topic": code_to_name.get(d["topic"], d["topic"]), "Within": d["amount"],
                           "Unit": d["unit"].replace("_", " "), "Late counts as": d["severity"],
                           "Source": d.get("source", "")} for d in prof.get("deadlines", [])],
                         columns=["Topic", "Within", "Unit", "Late counts as", "Source"])
    edited_dl = st.data_editor(
        dl_df, key="pf_deadlines", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"Topic": st.column_config.SelectboxColumn(options=list(name_to_code)),
                       "Within": st.column_config.NumberColumn(min_value=0.5, step=0.5),
                       "Unit": st.column_config.SelectboxColumn(options=["hours", "calendar days", "business days"]),
                       "Late counts as": st.column_config.SelectboxColumn(options=P.SEVERITIES)})

    st.subheader("Corrections")
    edited_corr = None
    if prof["corrections"]:
        st.caption("Tick **Approved** to let SCOPE learn from a correction. Delete a row to drop it.")
        names = {t["code"]: t["display"] for t in prof["topics"]}
        corr_df = pd.DataFrame([{
            "Approved": bool(c.get("approved")), "Date": c.get("date", ""), "By": c.get("by", ""),
            "Topic": names.get(c["issue"], c["issue"]),
            "Should be": (f"active, {c['severity']}" if c["status"] == "active" else c["status"].replace("_", " ")),
            "SCOPE said": c.get("scope_said", ""), "Quote": c["quote"], "Reason": c.get("reason", ""),
            "Visit risk": c.get("cra_risk") or ""} for c in prof["corrections"]])
        edited_corr = st.data_editor(
            corr_df, key="pf_corr", num_rows="dynamic", hide_index=True, width="stretch",
            disabled=[c for c in corr_df.columns if c != "Approved"])
    else:
        st.caption("No corrections yet. Use **Disagree with SCOPE? Correct it** under any result.")

    who = st.text_input("Your name or initials, for the change log", key="pf_who")
    b1, b2 = st.columns([1, 1])
    if b1.button("Save changes", type="primary"):
        new = json.loads(json.dumps(prof))
        new["name"] = name.strip() or prof["name"]
        rows = []
        for _, r in edited_topics.iterrows():
            display = str(r.get("Topic") or "").strip()
            if not display:
                continue
            code = str(r.get("Code") or "").strip() if isinstance(r.get("Code"), str) else ""
            rows.append({"code": code or P.code_for(display), "display": display,
                         "group": r.get("Group") or "Study-specific", "minor": r.get("Minor") or "",
                         "major": r.get("Major") or "", "critical": r.get("Critical") or "",
                         "enabled": bool(r.get("On")) if r.get("On") is not None else True})
        new["topics"] = rows
        new["study_rules"] = [x.strip() for x in rules.splitlines() if x.strip()]
        new["escalation"] = {"subjects_threshold": int(n_subj), "subjects_max": cap, "repeat": bool(rep_on)}
        new["thresholds"] = {"medium": int(med), "high": int(high)}
        new["deadlines"] = [
            {"topic": name_to_code.get(r["Topic"], r["Topic"]), "amount": r["Within"],
             "unit": str(r["Unit"] or "").replace(" ", "_"), "severity": r["Late counts as"] or "critical",
             "what": next((d.get("what", "") for d in prof.get("deadlines", [])
                           if code_to_name.get(d["topic"]) == r["Topic"]), ""),
             "source": r.get("Source") or "edited in SCOPE"}
            for _, r in edited_dl.iterrows() if r.get("Topic") and r.get("Within")]
        if edited_corr is not None:
            kept = []
            quotes = list(edited_corr["Quote"])
            approved = dict(zip(edited_corr["Quote"], edited_corr["Approved"]))
            for c in prof["corrections"]:
                if c["quote"] in quotes:
                    kept.append({**c, "approved": bool(approved[c["quote"]])})
            new["corrections"] = kept
        try:
            new = P.validate(new)
        except P.ProfileError as e:
            st.error(f"Not saved: {e}")
            return
        summary = _profile_diff(prof, new)
        if not summary:
            st.info("Nothing changed.")
            return
        if new["name"] == P.default_profile()["name"]:
            new["name"] = "SCOPE standard (customised)"  # the standard itself never changes silently
        new["version"] = P.bump_version(prof["version"])
        P.log_change(new, summary, who=who.strip())
        remember_study(new, replaces=prof["name"])
        st.success(f"Saved as version {new['version']}: {summary}. Saved under My studies (★) in this browser.")
        st.rerun()
    if b2.button("Start over from SCOPE standard"):
        st.session_state["profile"] = P.default_profile()
        st.rerun()

    if prof["changes"]:
        st.subheader("Change log")
        st.dataframe(pd.DataFrame(prof["changes"][::-1]), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
st.title("🩺 SCOPE")
st.caption("Site Communication & Oversight Processing Engine · reads free-text site-visit notes and returns a "
           "risk level, every finding with its evidence, the action items, and a draft follow-up letter")

def load_example_note() -> None:
    choice = st.session_state.get("example_pick")
    if choice:
        picked = note_choices()[choice]
        st.session_state["note"] = picked["text"]
        if "expected_risk" in picked:
            st.session_state["expected"] = picked
        else:
            st.session_state.pop("expected", None)


with st.sidebar:
    # 1. the study: protocol first, everything else is judged against it
    browser_sync()
    tracker_load()
    if learning_store() is not None:
        voter_id()  # ask the browser for its anonymous id early, so it is ready when someone rates or reviews
    if "pending_study" in st.session_state:  # a study was just created or saved: select it
        st.session_state["study_choice"] = st.session_state.pop("pending_study")
    if st.session_state.get("study_choice", GENERAL_STUDY) not in study_choices():  # e.g. browser data cleared
        st.session_state["study_choice"] = GENERAL_STUDY
    st.markdown("**1 · Which study?**")
    st.selectbox("Which study?", study_choices(), key="study_choice", on_change=select_study,
                 label_visibility="collapsed")
    demo = chosen_demo()
    choice = st.session_state.get("study_choice", GENERAL_STUDY)
    if choice.startswith(MY_PREFIX):
        mine = current_profile()
        n_rules = len(mine.get("protocol", {}).get("rules", [])) if mine.get("protocol") else len(mine["study_rules"])
        st.caption(f"Your study, saved in this browser · v{mine['version']} · {n_rules} protocol rules. Pick it any "
                   "time to score new visits; no need to upload the protocol again.")
    elif demo:
        st.caption(demo["summary"] + " (Fictional demo study.)")
        st.download_button("Read its protocol (PDF)", DM.protocol_path(demo["id"]).read_bytes(),
                           f"{demo['id']}-protocol.pdf", "application/pdf", width="stretch")
    elif st.session_state.get("study_choice") == OWN_STUDY:
        sidebar_protocol()
    else:
        st.caption("Scores notes with SCOPE's standard rubric. Pick a demo study or upload a protocol to score by a "
                   "study's own rules.")
    prof = current_profile()
    st.caption(f"Scoring with **{prof['name']}** (v{prof['version']}).")
    with st.expander("Saved study profiles"):
        up = st.file_uploader("Load a profile (.json)", type=["json"], key="profile_upload")
        if up is not None and st.session_state.get("profile_file_id") != up.file_id:
            st.session_state["profile_file_id"] = up.file_id
            try:
                remember_study(P.loads(up.getvalue().decode("utf-8")))
                st.rerun()
            except (P.ProfileError, UnicodeDecodeError) as e:
                st.error(f"Could not load that profile: {e}")
        st.download_button("Download this profile", P.dumps(prof), f"{prof['name']} v{prof['version']}.json",
                           "application/json", width="stretch")
        if choice.startswith(MY_PREFIX) and st.button("Remove this study from this browser", width="stretch"):
            my_studies().pop(choice[len(MY_PREFIX):], None)
            st.session_state["pending_study"] = GENERAL_STUDY
            st.session_state["profile"] = P.default_profile()
            st.rerun()
        st.caption("Your studies are kept in this browser. Download a copy to back it up, share it with a colleague, "
                   "or use it on another computer (load it above)." if _browser_js else
                   "Studies last for this browser session. Download yours to keep it, and load it next time.")
    st.divider()

    # 2. the AI engine
    st.markdown("**2 · AI engine**")
    engine = st.selectbox("Who reads the notes", ENGINE_CHOICES, key="byo_provider", label_visibility="collapsed")
    if engine == SHARED:
        st.caption(f"Free shared engine, {MAX_NEW_CALLS} new notes per session. Bring your own key for more, or to "
                   "use OpenAI, Claude or another model.")
    else:
        if engine.startswith("Other"):
            st.text_input("Base URL", key="byo_base", placeholder="https://api.mistral.ai/v1")
        st.text_input(f"Your {engine.split(' (')[0]} API key", type="password", key="byo_key",
                      help=KEY_HELP[engine] + " Kept only in this browser session; sent only to that provider.")
        e = engine_settings()
        models: list[str] = []
        if e["key"] or (engine.startswith("Other") and e["base"]):
            cache_id = hashlib.sha256(f"{engine}|{e['key']}|{e['base']}".encode()).hexdigest()
            if st.session_state.get("byo_models_for") != cache_id:
                try:
                    probe = PV.make_client(engine, e["key"], base_url=e["base"])
                    st.session_state["byo_models"] = PV.list_models(probe)
                except LLMError:
                    st.session_state["byo_models"] = []
                st.session_state["byo_models_for"] = cache_id
            models = st.session_state.get("byo_models", [])
        if models:
            st.selectbox("Model", models, key="byo_model")
        else:
            st.text_input("Model name", key="byo_model", placeholder="as the provider names it")
        st.caption(KEY_HELP[engine])
    st.divider()

    # 3. example notes
    st.markdown("**3 · Try a note**")
    st.selectbox("Example notes", list(note_choices()), index=None, placeholder="Choose an example note...",
                 key="example_pick", on_change=load_example_note, label_visibility="collapsed")
    st.caption("All notes, sites and people in this app are fictional. Only use fictional or de-identified notes: "
               "they are sent to the AI provider you choose.")
    with st.expander("How it works (TL;DR)"):
        st.markdown("An AI model reads the note. SCOPE keeps only findings it can match word for word in the note, "
                    "scores them with your study's rules, and flags anything it is unsure about for you to check. "
                    "When three people agree on a correction or a rating, SCOPE learns it, for everyone.")

parser = get_parser()
tab_one, tab_sites, tab_batch, tab_check, tab_prof, tab_learn = st.tabs(
    ["Analyze a note", "Sites & actions", "Portfolio view", "Accuracy check", "Study setup", "Help SCOPE learn"])

with tab_one:
    if not st.session_state.get("note", "").strip():
        st.info("**New here?** Pick a demo study and one of its notes in the sidebar: two clicks, then SCOPE reads it. "
                "Have a minute more? Open **Help SCOPE learn** and rate a practice visit; every rating teaches SCOPE.")
    note = st.text_area("Site-visit note", key="note", height=230,
                        placeholder="Paste a monitoring visit report, field note or visit e-mail...")
    rec = None
    if not parser:
        st.info("SCOPE needs a Gemini API key to read notes. Paste a free key from aistudio.google.com in the "
                "sidebar, or add `GEMINI_API_KEY` to the app's secrets.")
    elif note.strip():
        try:
            with st.spinner("Reading the note (usually under 30 seconds)..."):
                rec = cached_llm("record", note, lambda: parser.analyze(note))
        except LLMError as e:
            show_llm_error(e)
            om = own_model()
            if om is not None and om.active:
                guess = om.predict(note)
                st.info("While the AI reading is unavailable, **SCOPE's own model** estimates "
                        f"**{guess['risk'].upper()}** risk for this visit. It is a rough estimate of the risk level "
                        f"only (it agrees with experts on {om.report['accuracy']:.0%} of notes), with no findings or "
                        "evidence. Try the full reading again later.")
    if rec:
        v = rec["visit"]
        c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
        c1.markdown(risk_badge(rec["risk"]["level"]), unsafe_allow_html=True)
        c1.caption(risk_reason(rec))
        c2.metric("Site", v["site"]["id"] or "-")
        c3.metric("Visit date", v["visit_date"]["iso"] or (v["visit_date"]["text"] or "-"))
        c4.metric("Visit type", v["visit_type"]["code"] or "-")
        for alert in rec.get("alerts", []):
            st.error("**Safety check:** " + alert)
        for check in rec.get("checks", []):
            st.info("**Deadline check:** " + check)
        expected = st.session_state.get("expected")
        if expected and expected.get("text") == note:
            agrees = expected["expected_risk"] == rec["risk"]["level"]
            st.caption(f"{'Matches' if agrees else 'Differs from'} what a reviewer would expect for this demo note: "
                       f"**{expected['expected_risk'].upper()}**. {expected['why']}")
        if rec.get("summary"):
            st.markdown(f"> {rec['summary']}")
        others = [r for r in rec["review"]["reasons"] if r not in rec.get("alerts", [])]
        if others:
            st.warning("**Check before relying on this:** " + "; ".join(others) + ".")

        site_history_panel(rec, note)
        t_find, t_note, t_act, t_report, t_letter, t_sim, t_audit, t_json = st.tabs(
            ["Findings", "Highlighted note", "Visit details & actions", "Visit report", "Follow-up letter",
             "Similar past visits", "Audit summary", "JSON"])
        with t_find:
            active = findings_table(rec, "active")
            st.subheader(f"Active findings ({len(active)})")
            if len(active):
                st.dataframe(active, hide_index=True, width="stretch")
            else:
                st.success("No active problems in this note.")
            fixed = findings_table(rec, "resolved_on_site")
            if len(fixed):
                st.subheader(f"Fixed during the visit ({len(fixed)})")
                st.dataframe(fixed, hide_index=True, width="stretch")
            fine = findings_table(rec, "no_issue")
            if len(fine):
                with st.expander(f"Checked and fine ({len(fine)})"):
                    st.dataframe(fine, hide_index=True, width="stretch")
            ignored = [f for f in rec.get("findings", []) if not f.get("verified")]
            if ignored:
                with st.expander(f"Ignored: quote not found in the note ({len(ignored)})"):
                    st.dataframe(pd.DataFrame([{"Issue": f["display"], "Status": f["status"],
                                                "Claimed quote": f["evidence"]} for f in ignored]),
                                 hide_index=True, width="stretch")
            prof_used = rec.get("profile") or {}
            took = f" in {rec['seconds']:.0f} s" if rec.get("seconds") else ""
            st.caption(f"Read by {rec.get('provider', 'Google Gemini')} {rec.get('model') or ''}{took}; risk computed "
                       "from the verified active findings "
                       f"with the study profile {prof_used.get('name', 'SCOPE standard')} "
                       f"(v{prof_used.get('version', '3.1')}).")
            second = OM.second_opinion(own_model(), note, rec["risk"]["level"])
            if second and second["second_look"]:
                st.warning("**Second look:** SCOPE's own model, trained on notes experts labelled, reads this visit as "
                           "**HIGH** risk; the AI reading says LOW. Check the note for a problem the reading missed.")
            elif second:
                st.caption(f"SCOPE's own model reads this visit as {second['risk'].upper()} risk.")
            learned = rec.get("learned_from") or []
            if learned:
                st.caption(f"SCOPE used {len(learned)} ruling{'s' if len(learned) > 1 else ''} it learned from "
                           "reviewers on similar notes (see How this was read).")
            if rec.get("engine_log") or learned:
                with st.expander("How this was read"):
                    st.markdown("\n".join(f"- {line}" for line in rec.get("engine_log", [])))
                    if learned:
                        st.markdown("**Rulings learned from reviewers, used for this note:**\n" + "\n".join(
                            f"- {x['ruling']}" for x in learned))
            feedback(note, rec)
        with t_note:
            st.markdown(highlight(note, rec), unsafe_allow_html=True)
        with t_act:
            st.markdown(
                f"- **Visit type:** {v['visit_type']['name'] or v['visit_type']['text'] or '-'}  \n"
                f"- **Monitor:** {v['monitor'] or '-'}  \n- **PI:** {v['pi'] or '-'}  \n"
                f"- **Screened / enrolled:** {v['screened'] if v['screened'] is not None else '-'} / "
                f"{v['enrolled'] if v['enrolled'] is not None else '-'}")
            st.subheader("Open action items")
            if rec["actions"]:
                st.dataframe(pd.DataFrame([{"Action": a["action"], "Owner": a["owner"] or "-",
                                            "Due": a["due_date"] or a["due"] or "-"} for a in rec["actions"]]),
                             hide_index=True, width="stretch")
            else:
                st.info("No open action items in the note.")
            for w in rec["warnings"]:
                st.caption(f"Note: {w}")
        with t_report:
            st.caption("Turn rough notes into a full visit report: SCOPE writes it from the note and its verified "
                       "reading only, and leaves anything the note does not say as a [placeholder] for you.")
            if st.button("Draft visit report"):
                try:
                    with st.spinner("Drafting the report..."):
                        st.session_state["report"] = (note, cached_llm(
                            "report", note, lambda: RP.draft_report(parser.client, rec, note)))
                except LLMError as e:
                    st.error(str(e))
            report = st.session_state.get("report")
            if report and report[0] == note:
                edited_report = st.text_area("Report draft (edit before filing)", report[1], height=520)
                st.download_button("Download report (.md)", edited_report, "visit_report.md", "text/markdown")
                missing = RP.missing_sections(edited_report)
                holes = RP.placeholders(edited_report)
                if holes:
                    st.caption("To fill in: " + ", ".join(holes[:12]) + ("…" if len(holes) > 12 else ""))
                if missing:
                    st.warning("The draft is missing: " + ", ".join(missing) + ". Add them or draft again.")
        with t_letter:
            if st.button("Draft follow-up letter"):
                try:
                    with st.spinner("Drafting..."):
                        st.session_state["letter"] = (note, cached_llm(
                            "letter", note, lambda: draft_followup(parser.client, rec)))
                except LLMError as e:
                    st.error(str(e))
            letter = st.session_state.get("letter")
            if letter and letter[0] == note:
                edited = st.text_area("Draft (edit before sending)", letter[1], height=420)
                st.download_button("Download letter (.md)", edited, "follow_up_letter.md", "text/markdown")
                st.caption("Built only from the verified findings and action items; missing details are left as "
                           "[placeholders].")
        with t_sim:
            index = get_index()
            codes = [i["code"] for i in rec["issues"]]
            only_shared = st.checkbox("Only show visits that share an active issue", value=bool(codes))
            hits = index.search(note, k=5, issue_filter=codes if only_shared else None)
            st.caption(f"Keyword search over {len(index.rows)} past (synthetic) visits.")
            for h in hits:
                m = h["meta"]
                issues = ", ".join(ISSUE_BY_CODE[c].display for c in h["issues"]) or "no active issues"
                with st.expander(f"Site {m.get('site_id', '?')} · {m.get('visit_date', '?')} · {h['risk']} risk · "
                                 f"{issues}"):
                    st.text(h["text"])
        with t_audit:
            md = audit_summary(rec)
            st.markdown(md)
            st.download_button("Download summary (.md)", md, "scope_visit_summary.md", "text/markdown")
        with t_json:
            st.caption("SCOPE's record. `llm_output` is exactly what the LLM returned, before SCOPE checked it.")
            st.json(rec)
            st.download_button("Download record (.json)", json.dumps(rec, indent=2), "scope_visit_record.json",
                               "application/json")

with tab_batch:
    st.markdown(f"Run SCOPE over several visits and see where the risk is. Upload a CSV with a `note` column "
                f"(first {PORTFOLIO_MAX} rows), or use a sample of past (synthetic) visits.")
    up = st.file_uploader("CSV with a 'note' column", type=["csv"])
    texts: list[str] = []
    if up is not None:
        df_in = pd.read_csv(up)
        if "note" not in df_in.columns:
            st.error("The CSV needs a column named 'note'.")
        else:
            texts = df_in["note"].astype(str).tolist()[:PORTFOLIO_MAX]
    elif st.button(f"Analyze {PORTFOLIO_MAX} sample visits"):
        texts = [r["text"] for r in read_jsonl(CORPUS)[:PORTFOLIO_MAX]]
    if texts and not parser:
        st.info("Add a Gemini API key in the sidebar first.")
    elif texts:
        recs, bar = [], st.progress(0.0, text="Reading notes...")
        skipped = 0
        for i, t in enumerate(texts):
            try:
                recs.append(cached_llm("record", t, lambda t=t: parser.analyze(t)))
            except Unreadable:
                skipped += 1
            except LLMError as e:
                st.error(f"Stopped after {i} notes: {e}")
                break
            bar.progress((i + 1) / len(texts), text=f"Read {i + 1} of {len(texts)} notes")
        if skipped:
            st.warning(f"{skipped} note(s) could not be read reliably and are left out. Run them again later.")
        if recs:
            df = pd.DataFrame([to_row(r) for r in recs])
            c1, c2, c3 = st.columns(3)
            c1.metric("Visits", len(df))
            c2.metric("High risk", int((df["risk"] == "high").sum()),
                      help=f"{int(df['safety_alert'].sum())} visit(s) also have a safety alert to check by hand")
            c3.metric("Open action items", int(df["n_actions"].sum()))
            counts = pd.DataFrame([{"issue": i["display"], "group": i["group"]} for r in recs for i in r["issues"]])
            if not counts.empty:
                st.altair_chart(alt.Chart(counts).mark_bar().encode(
                    x=alt.X("count():Q", title="Visits with this issue"),
                    y=alt.Y("issue:N", sort="-x", title=None),
                    color=alt.Color("group:N", title="Group", scale=alt.Scale(domain=ISSUE_GROUPS)),
                ).properties(height=300, title="Active issues across visits"), width="stretch")
            order = {"high": 0, "medium": 1, "low": 2}
            st.dataframe(df.sort_values("risk", key=lambda s: s.map(order)), hide_index=True, width="stretch")
            st.download_button("Download table (.csv)", df.to_csv(index=False), "scope_portfolio.csv", "text/csv")

with tab_check:
    st.markdown(
        "Does SCOPE agree with expert judgement? Run it on notes that already have expert labels (risk "
        "level and "
        "active issues) and compare. None of these notes are in SCOPE's instructions, so it has not seen the answers. "
        "The stress-test notes were written to cover many styles, all 22 issue types and common traps; the other sets "
        "helped shape the rubric.")

    @st.cache_data
    def labelled_sets() -> dict[str, list[dict]]:
        hw = load_handwritten()
        st_notes = load_stress()
        return {"Stress test notes 1-13": st_notes[:13], "Stress test notes 14-25": st_notes[13:25],
                "Formal visit reports (7)": load_realistic(), "Hand-written notes 1-8": hw[:8],
                "Hand-written notes 9-16": hw[8:16], "Hand-written notes 17-24": hw[16:24]}

    sets = {f"Demo studies ({len(DM.labelled_rows())} notes, each under its own protocol)": DM.labelled_rows(),
            **labelled_sets()}
    lib_now = library()
    agreed = L.community_rows(lib_now, practice_notes()) if lib_now else []
    if agreed:
        sets[f"Practice notes rated by the community ({len(agreed)} agreed)"] = agreed
    corrected = [c for c in current_profile()["corrections"] if c.get("cra_risk") and c.get("note")]
    if corrected:
        unique = list({c["note"]: c for c in corrected}.values())
        sets[f"Notes corrected in this profile ({len(unique[:10])})"] = [
            {"id": f"corr-{i + 1}", "text": c["note"], "risk": c["cra_risk"], "issues": [], "risk_only": True}
            for i, c in enumerate(unique[:10])]
    choice = st.selectbox("Labelled notes", list(sets))
    if not parser:
        st.info("Add a Gemini API key in the sidebar first.")
    elif st.button("Run the check"):
        st.session_state["check"] = choice
    if parser and st.session_state.get("check") == choice:
        rows, recs, bar = sets[choice], [], st.progress(0.0, text="Reading notes...")
        unread: dict[str, Unreadable] = {}
        for i, r in enumerate(rows):
            try:
                recs.append(analyze_under(r["text"], DM.profile(r["study"]) if r.get("study") else None))
            except Unreadable as e:  # this note could not be read; carry on with the others
                recs.append(None)
                unread[r["id"]] = e
            except LLMError as e:  # quota or busy: stop
                st.error(f"Stopped after {i} notes: {e}")
                break
            bar.progress((i + 1) / len(rows), text=f"Read {i + 1} of {len(rows)} notes")
        for note_id, e in unread.items():
            with st.expander(f"{note_id}: not scored, Gemini's answer was unusable ({e.reason}, {e.model})"):
                st.code(e.raw[:6000] or "(empty)")
        if any(recs):
            topic_names = {t["code"]: t["display"] for t in current_profile()["topics"]}

            def names(codes):
                return ", ".join(topic_names.get(c) or (ISSUE_BY_CODE[c].display if c in ISSUE_BY_CODE else c)
                                 for c in sorted(codes)) or "none"

            table = []
            for r, rec in zip(rows, recs):
                if rec is None:
                    table.append({"Note": r["id"], "Starts with": " ".join(r["text"].split())[:70] + "...",
                                  "Expert risk": r["risk"], "SCOPE risk": "not read", "Risk agrees": "-",
                                  "Expert issues": names(r["issues"]), "SCOPE issues": "-", "Issues agree": "-",
                                  "Safety alert": ""})
                    continue
                got = {i["code"] for i in rec["issues"]}
                risk_only = r.get("risk_only")
                table.append({"Note": r["id"], "Starts with": " ".join(r["text"].split())[:70] + "...",
                              "Expert risk": r["risk"], "SCOPE risk": rec["risk"]["level"],
                              "Risk agrees": "yes" if r["risk"] == rec["risk"]["level"] else "NO",
                              "Expert issues": "-" if risk_only else names(r["issues"]), "SCOPE issues": names(got),
                              "Issues agree": "-" if risk_only else ("yes" if set(r["issues"]) == got else "partly"),
                              "Safety alert": "yes" if rec.get("alerts") else ""})
            df = pd.DataFrame(table)
            n = int((df["SCOPE risk"] != "not read").sum())
            high = df[df["Expert risk"] == "high"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk level agrees", f"{int((df['Risk agrees'] == 'yes').sum())} / {n}")
            c2.metric("High-risk visits caught", f"{int((high['SCOPE risk'] == 'high').sum())} / {len(high)}"
                      if len(high) else "none in set")
            c3.metric("False alarms (flagged high, expert said lower)",
                      int(((df["SCOPE risk"] == "high") & (df["Expert risk"] != "high")).sum()))
            st.dataframe(df, hide_index=True, width="stretch")
            for r, rec in zip(rows, recs):
                if rec is None or r["risk"] == rec["risk"]["level"]:
                    continue
                with st.expander(f"{r['id']}: expert said {r['risk']}, SCOPE said {rec['risk']['level']}"):
                    if r.get("why"):
                        st.caption(f"Expected: {r['why']}")
                    st.text(r["text"])
                    st.dataframe(findings_table(rec, "active"), hide_index=True, width="stretch")
                    st.caption("Who is right? If the label looks wrong to you, tell us; if SCOPE is wrong, this "
                               "is what the next prompt or rubric change should fix.")

with tab_prof:
    profile_editor()

with tab_learn:
    learned_tab()

with tab_sites:
    sites_tab()

tracker_save()  # last, so this run's changes to the site history are written to the browser
