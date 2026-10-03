"""SCOPE Streamlit front end (LLM engine).

    export GEMINI_API_KEY=...        # or add it to .streamlit/secrets.toml / the app's Secrets
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
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
from scope import protocol as PR  # noqa: E402
from scope import providers as PV  # noqa: E402
from scope.data.generate import read_jsonl  # noqa: E402
from scope.data.handwritten import load_handwritten, load_realistic, load_stress  # noqa: E402
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
    """The study profile in use in this browser session (default: SCOPE standard, rubric v3.1)."""
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
    return parser


@st.cache_resource
def shared_cache() -> dict:
    """Answers shared by all visitors for 24 hours, so repeated example notes cost no quota."""
    return {}


def _cache_key(kind: str, text: str) -> str:
    fp = hashlib.sha256((api_key() or "").encode()).hexdigest()[:12]
    prof = P.fingerprint(current_profile())
    return f"{kind}:{ENGINE_VERSION}:{prof}:{fp}:{hashlib.sha256(text.encode()).hexdigest()}"


def cached_llm(kind: str, text: str, fn):
    cache, key = shared_cache(), _cache_key(kind, text)
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
    if isinstance(e, Unreadable) and e.raw:
        with st.expander("What Gemini returned (not used)"):
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
             "Raised because": ", ".join(f.get("escalated_by") or []),
             "Evidence (quoted from the note)": f["evidence"], "Why": f.get("explanation", "")}
            for f in rec.get("findings", []) if f["status"] == status and f.get("verified")]
    order = {"critical": 0, "major": 1, "minor": 2}
    rows.sort(key=lambda r: order.get(r["Severity"], 3))
    if status != "active":
        for r in rows:
            r.pop("Severity")
            r.pop("Raised because")
    elif not any(r["Raised because"] for r in rows):
        for r in rows:
            r.pop("Raised because")
    return pd.DataFrame(rows)


def correction_form(note: str, rec: dict) -> None:
    """Let a CRA correct SCOPE. Corrections are proposed; a lead CRA approves them in the Study profile tab."""
    prof = current_profile()
    topics = P.enabled_topics(prof)
    found = [f for f in rec.get("findings", []) if f.get("verified")]
    with st.expander("Disagree with SCOPE? Correct it"):
        st.caption("Your correction is saved to the study profile as a proposal. Once a lead CRA approves it, SCOPE "
                   "shows it to the LLM as an example whenever it reads a similar note.")
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
                st.success("Correction saved as a proposal. Approve it in the Study profile tab, then download the "
                           "profile to keep it.")


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
    if old["thresholds"] != new["thresholds"]:
        parts.append(f"risk thresholds medium {new['thresholds']['medium']}, high {new['thresholds']['high']}")
    oa = sum(bool(c.get("approved")) for c in old["corrections"])
    na = sum(bool(c.get("approved")) for c in new["corrections"])
    if (oa, len(old["corrections"])) != (na, len(new["corrections"])):
        parts.append(f"corrections: {na} approved of {len(new['corrections'])}")
    if old["name"] != new["name"]:
        parts.append(f"renamed to {new['name']}")
    return "; ".join(parts)


EXAMPLE_PROTOCOL = ROOT / "profiles" / "example_protocol_ZLV-301.pdf"
EXAMPLE_PROFILE = ROOT / "profiles" / "example_oncology_study.json"


def protocol_builder() -> None:
    """Upload a protocol, let the LLM draft its study rules (quotes checked), and let a lead CRA accept them."""
    st.subheader("Build a profile from a protocol")
    st.markdown(
        "SCOPE reads the protocol and drafts the rules that change how visits should be judged: what counts as an "
        "SAE and its reporting deadline, visit windows, key eligibility criteria, dosing and storage rules, and what "
        "the protocol calls an important deviation. Every rule comes with the exact quote and page, checked against "
        "the protocol. Nothing is used until you accept it.")
    st.warning("Free-tier requests may be used by the AI provider. Only upload protocols that are public (for "
               "example from ClinicalTrials.gov) or fictional. Never upload a confidential sponsor protocol here.")
    presets = {"SCOPE standard (rubric v3.1)": "standard", "The current profile": "current"}
    if EXAMPLE_PROFILE.exists():
        presets["Example oncology profile"] = "example"
    c1, c2 = st.columns([1, 1])
    base_choice = c1.selectbox("Start from", list(presets), key="pr_base")
    upload = c2.file_uploader("Protocol (PDF, Word or text)", type=["pdf", "docx", "txt"], key="pr_file")
    use_example = EXAMPLE_PROTOCOL.exists() and st.button("Use the example protocol (ZLV-301, fictional)")
    source = None
    if upload is not None:
        source = (upload.name, upload.getvalue())
    elif use_example:
        source = (EXAMPLE_PROTOCOL.name, EXAMPLE_PROTOCOL.read_bytes())
    if source and (use_example or st.button("Read the protocol", type="primary")):
        if not parser:
            st.info("Add a Gemini API key in the sidebar first.")
            return
        base = {"standard": P.default_profile, "current": current_profile,
                "example": lambda: P.loads(EXAMPLE_PROFILE.read_text())}[presets[base_choice]]()
        try:
            pages = PR.read_document(*source)
            key = f"{PR.fingerprint(source[1])}:{P.fingerprint(base)}"
            with st.spinner("Reading the protocol..."):
                draft = cached_llm("protocol", key, lambda: PR.draft_rules(parser.client, pages, base))
            st.session_state["protocol_draft"] = {"file": source[0], "draft": draft, "base": base}
        except LLMError as e:
            show_llm_error(e)
    pd_state = st.session_state.get("protocol_draft")
    if not pd_state:
        return
    draft, base = pd_state["draft"], pd_state["base"]
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
        st.session_state["profile"] = new
        st.session_state.pop("protocol_draft", None)
        st.success(f"Created {new['name']} v{new['version']} with {len(accepted)} protocol rules. Every visit is now "
                   "scored against this protocol. Download the profile to keep it.")
        st.rerun()


def profile_editor() -> None:
    prof = current_profile()
    st.markdown(
        "A study profile is SCOPE's rubric for one study. Start from **SCOPE standard** (rubric v3.1, written by an "
        "experienced CRA), then adjust it to the protocol: change what counts as minor, major or critical, add "
        "study-specific topics and rules, tune escalation, and approve CRA corrections so SCOPE learns from them.")
    st.info("Changes last for this browser session. Use **Download this profile** in the sidebar to keep them, and "
            "load the file next time. Every result shows which profile and version scored it.")
    protocol_builder()
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

    st.subheader("CRA corrections")
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
        st.session_state["profile"] = new
        st.success(f"Saved as version {new['version']}: {summary}. Download the profile to keep it.")
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

with st.sidebar:
    st.markdown("**Example notes**")
    for label, text in examples().items():
        if st.button(label, width="stretch"):
            st.session_state["note"] = text
    st.caption("All notes, sites and people in this app are fictional.")
    st.divider()
    st.markdown("**AI engine** (SCOPE checks and scores whatever it reads)")
    engine = st.selectbox("Who reads the notes", ENGINE_CHOICES, key="byo_provider", label_visibility="collapsed")
    if engine == SHARED:
        st.caption(f"The app's shared free Gemini quota ({MAX_NEW_CALLS} new notes per session). Bring your own key "
                   "for more, or to use OpenAI, Claude or another model.")
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
    st.caption("Only use fictional or de-identified notes: they are sent to the AI provider you choose.")
    st.divider()
    prof = current_profile()
    st.markdown(f"**Study profile:** {prof['name']} (v{prof['version']})")
    up = st.file_uploader("Load a study profile (.json)", type=["json"], key="profile_upload")
    if up is not None and st.session_state.get("profile_file_id") != up.file_id:
        st.session_state["profile_file_id"] = up.file_id
        try:
            st.session_state["profile"] = P.loads(up.getvalue().decode("utf-8"))
            st.rerun()
        except (P.ProfileError, UnicodeDecodeError) as e:
            st.error(f"Could not load that profile: {e}")
    if EXAMPLE_PROFILE.exists() and st.button("Try the example oncology profile", width="stretch"):
        st.session_state["profile"] = P.loads(EXAMPLE_PROFILE.read_text())
        st.rerun()
    st.download_button("Download this profile", P.dumps(prof), f"{prof['name']} v{prof['version']}.json",
                       "application/json", width="stretch")
    st.caption("Profiles live in this browser session. Download yours to keep it, and load it next time.")
    with st.expander("How SCOPE decides"):
        st.markdown(
            "1. The LLM reads the note and lists every topic it mentions: an **active** problem, something "
            "**fixed during the visit**, or **confirmed fine**, each with a quote from the note.\n"
            "2. SCOPE checks every quote, name, date and action against the note and drops anything that is "
            "not there.\n"
            "3. SCOPE scores with the study profile's rubric. The default (SCOPE standard, 22 topics): any "
            "critical finding or two major findings = high; one major or three minor = medium; otherwise low. A "
            "repeat finding is raised one level, and so is a problem affecting 3 or more subjects (up to major). "
            "A study profile can change topics, severities, study rules, escalation and thresholds.\n"
            "4. CRA corrections that a lead CRA approves are shown to the LLM as examples for similar notes, so "
            "SCOPE adapts to the study without retraining.\n"
            "5. Safety net: an empty or garbled answer from the LLM is asked again and never scored, and if the note "
            "mentions a possible SAE, consent problem, dosing error or IRB lapse that the LLM did not report, SCOPE "
            "shows a red safety alert.")

parser = get_parser()
tab_one, tab_batch, tab_check, tab_prof = st.tabs(["Analyze a note", "Portfolio view", "Accuracy check",
                                                   "Study profile"])

with tab_one:
    note = st.text_area("Site-visit note", key="note", height=230,
                        placeholder="Paste a monitoring visit report, field note or visit e-mail...")
    rec = None
    if not parser:
        st.info("SCOPE needs a Gemini API key to read notes. Paste a free key from aistudio.google.com in the "
                "sidebar, or add `GEMINI_API_KEY` to the app's secrets.")
    elif note.strip():
        try:
            with st.spinner("Reading the note..."):
                rec = cached_llm("record", note, lambda: parser.analyze(note))
        except LLMError as e:
            show_llm_error(e)
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
        if rec.get("summary"):
            st.markdown(f"> {rec['summary']}")
        others = [r for r in rec["review"]["reasons"] if r not in rec.get("alerts", [])]
        if others:
            st.warning("**Check before relying on this:** " + "; ".join(others) + ".")

        t_find, t_note, t_act, t_letter, t_sim, t_audit, t_json = st.tabs(
            ["Findings", "Highlighted note", "Visit details & actions", "Follow-up letter", "Similar past visits",
             "Audit summary", "JSON"])
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
            st.caption(f"Read by {rec.get('provider', 'Google Gemini')} {rec.get('model') or ''}; risk computed "
                       "from the verified active findings "
                       f"with the study profile {prof_used.get('name', 'SCOPE standard')} "
                       f"(v{prof_used.get('version', '3.1')}).")
            correction_form(note, rec)
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
        "Does SCOPE agree with an experienced CRA? Run it on notes that a CRA has already labelled (risk level and "
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

    sets = dict(labelled_sets())
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
                recs.append(cached_llm("record", r["text"], lambda t=r["text"]: parser.analyze(t)))
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
                                  "CRA risk": r["risk"], "SCOPE risk": "not read", "Risk agrees": "-",
                                  "CRA issues": names(r["issues"]), "SCOPE issues": "-", "Issues agree": "-",
                                  "Safety alert": ""})
                    continue
                got = {i["code"] for i in rec["issues"]}
                risk_only = r.get("risk_only")
                table.append({"Note": r["id"], "Starts with": " ".join(r["text"].split())[:70] + "...",
                              "CRA risk": r["risk"], "SCOPE risk": rec["risk"]["level"],
                              "Risk agrees": "yes" if r["risk"] == rec["risk"]["level"] else "NO",
                              "CRA issues": "-" if risk_only else names(r["issues"]), "SCOPE issues": names(got),
                              "Issues agree": "-" if risk_only else ("yes" if set(r["issues"]) == got else "partly"),
                              "Safety alert": "yes" if rec.get("alerts") else ""})
            df = pd.DataFrame(table)
            n = int((df["SCOPE risk"] != "not read").sum())
            high = df[df["CRA risk"] == "high"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk level agrees", f"{int((df['Risk agrees'] == 'yes').sum())} / {n}")
            c2.metric("High-risk visits caught", f"{int((high['SCOPE risk'] == 'high').sum())} / {len(high)}"
                      if len(high) else "none in set")
            c3.metric("False alarms (flagged high, CRA said lower)",
                      int(((df["SCOPE risk"] == "high") & (df["CRA risk"] != "high")).sum()))
            st.dataframe(df, hide_index=True, width="stretch")
            for r, rec in zip(rows, recs):
                if rec is None or r["risk"] == rec["risk"]["level"]:
                    continue
                with st.expander(f"{r['id']}: CRA said {r['risk']}, SCOPE said {rec['risk']['level']}"):
                    st.text(r["text"])
                    st.dataframe(findings_table(rec, "active"), hide_index=True, width="stretch")
                    st.caption("Who is right? If the label looks wrong to you, tell us; if SCOPE is wrong, this "
                               "is what the next prompt or rubric change should fix.")

with tab_prof:
    profile_editor()
