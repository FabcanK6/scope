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

from scope.data.generate import read_jsonl  # noqa: E402
from scope.data.handwritten import load_handwritten, load_realistic  # noqa: E402
from scope.engine import SYSTEM, LLMParser  # noqa: E402
from scope.llm import GeminiClient, LLMError, draft_followup, get_api_key  # noqa: E402
from scope.record import audit_summary, to_row  # noqa: E402
from scope.schema import ISSUE_BY_CODE, ISSUE_GROUPS  # noqa: E402

CORPUS = ROOT / "data" / "corpus.jsonl"
RISK_COLORS = {"low": "#2E7D32", "medium": "#ED6C02", "high": "#C62828"}
SEVERITY_COLORS = {"critical": "#C62828", "major": "#ED6C02", "minor": "#B8860B"}
STATUS_COLORS = {"resolved_on_site": "#1565C0", "no_issue": "#2E7D32"}
MAX_NEW_CALLS = 25  # new LLM requests per browser session (cached answers are free)
PORTFOLIO_MAX = 10
ENGINE_VERSION = hashlib.sha256(SYSTEM.encode()).hexdigest()[:8]  # a new prompt never reuses old cached answers

st.set_page_config(page_title="SCOPE - Site Visit Note Intelligence", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# LLM access and caching
# ---------------------------------------------------------------------------
def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets configured
        return None


def api_key() -> str | None:
    return st.session_state.get("byo_key") or get_api_key(st.secrets if _secret("GEMINI_API_KEY") else None)


def get_parser() -> LLMParser | None:
    key = api_key()
    if not key:
        return None
    parser = st.session_state.get("parser")
    if parser is None or parser.client.api_key != key:
        parser = LLMParser(GeminiClient(key, model=_secret("GEMINI_MODEL")))
        st.session_state["parser"] = parser
    return parser


@st.cache_resource
def shared_cache() -> dict:
    """Answers shared by all visitors for 24 hours, so repeated example notes cost no quota."""
    return {}


def _cache_key(kind: str, text: str) -> str:
    fp = hashlib.sha256((api_key() or "").encode()).hexdigest()[:12]
    return f"{kind}:{ENGINE_VERSION}:{fp}:{hashlib.sha256(text.encode()).hexdigest()}"


def cached_llm(kind: str, text: str, fn):
    cache, key = shared_cache(), _cache_key(kind, text)
    hit = cache.get(key)
    if hit and time.time() - hit[0] < 24 * 3600:
        return hit[1]
    used = st.session_state.get("llm_calls", 0)
    if used >= MAX_NEW_CALLS:
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
            color = SEVERITY_COLORS[f["severity"]] if f["status"] == "active" else STATUS_COLORS[f["status"]]
            label = f"{f['display']} · {f['severity'] if f['status'] == 'active' else f['status'].replace('_', ' ')}"
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
    rows = [{"Issue": f["display"], "Severity": f["severity"], "Evidence (quoted from the note)": f["evidence"],
             "Why": f.get("explanation", "")}
            for f in rec.get("findings", []) if f["status"] == status and f.get("verified")]
    order = {"critical": 0, "major": 1, "minor": 2}
    rows.sort(key=lambda r: order.get(r["Severity"], 3))
    if status != "active":
        for r in rows:
            r.pop("Severity")
    return pd.DataFrame(rows)


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
    st.markdown("**Engine:** Google Gemini, checked by SCOPE")
    if _secret("GEMINI_API_KEY") and not st.session_state.get("byo_key"):
        st.caption("Using the app's shared free quota.")
    st.text_input("Your own Gemini API key (optional)", type="password", key="byo_key",
                  help="Free key from aistudio.google.com. Kept only in this browser session.")
    st.caption("Notes are sent to Google's Gemini API. Only use fictional or de-identified notes.")
    with st.expander("How SCOPE decides"):
        st.markdown(
            "1. The LLM reads the note and lists every topic it mentions: an **active** problem, something "
            "**fixed during the visit**, or **confirmed fine**, each with a quote from the note.\n"
            "2. SCOPE checks every quote, name, date and action against the note and drops anything that is "
            "not there.\n"
            "3. SCOPE applies the severity rubric: any critical finding or two major findings = high; one major "
            "or three minor = medium; otherwise low. Late or unreported SAEs, consent after procedures and "
            "dosing errors are critical.")

parser = get_parser()
tab_one, tab_batch, tab_check = st.tabs(["Analyze a note", "Portfolio view", "Accuracy check"])

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
            st.error(str(e))
    if rec:
        v = rec["visit"]
        c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
        c1.markdown(risk_badge(rec["risk"]["level"]), unsafe_allow_html=True)
        c1.caption(risk_reason(rec))
        c2.metric("Site", v["site"]["id"] or "-")
        c3.metric("Visit date", v["visit_date"]["iso"] or (v["visit_date"]["text"] or "-"))
        c4.metric("Visit type", v["visit_type"]["code"] or "-")
        if rec.get("summary"):
            st.markdown(f"> {rec['summary']}")
        if rec["review"]["needed"]:
            st.warning("**Check before relying on this:** " + "; ".join(rec["review"]["reasons"]) + ".")

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
            st.caption(f"Read by {rec.get('model') or 'Gemini'}; risk computed by SCOPE's rubric from the verified "
                       "active findings.")
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
        for i, t in enumerate(texts):
            try:
                recs.append(cached_llm("record", t, lambda t=t: parser.analyze(t)))
            except LLMError as e:
                st.error(f"Stopped after {i} notes: {e}")
                break
            bar.progress((i + 1) / len(texts), text=f"Read {i + 1} of {len(texts)} notes")
        if recs:
            df = pd.DataFrame([to_row(r) for r in recs])
            c1, c2, c3 = st.columns(3)
            c1.metric("Visits", len(df))
            c2.metric("High risk", int((df["risk"] == "high").sum()))
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
        "active issues) and compare. None of these notes are in SCOPE's instructions, so it has not seen the answers.")

    @st.cache_data
    def labelled_sets() -> dict[str, list[dict]]:
        hw = load_handwritten()
        return {"Formal visit reports (7)": load_realistic(), "Hand-written notes 1-8": hw[:8],
                "Hand-written notes 9-16": hw[8:16], "Hand-written notes 17-24": hw[16:24]}

    sets = labelled_sets()
    choice = st.selectbox("Labelled notes", list(sets))
    if not parser:
        st.info("Add a Gemini API key in the sidebar first.")
    elif st.button("Run the check"):
        st.session_state["check"] = choice
    if parser and st.session_state.get("check") == choice:
        rows, recs, bar = sets[choice], [], st.progress(0.0, text="Reading notes...")
        for i, r in enumerate(rows):
            try:
                recs.append(cached_llm("record", r["text"], lambda t=r["text"]: parser.analyze(t)))
            except LLMError as e:
                st.error(f"Stopped after {i} notes: {e}")
                break
            bar.progress((i + 1) / len(rows), text=f"Read {i + 1} of {len(rows)} notes")
        if recs:
            def names(codes):
                return ", ".join(ISSUE_BY_CODE[c].display for c in sorted(codes)) or "none"

            table = []
            for r, rec in zip(rows, recs):
                got = {i["code"] for i in rec["issues"]}
                table.append({"Note": r["id"], "Starts with": " ".join(r["text"].split())[:70] + "...",
                              "CRA risk": r["risk"], "SCOPE risk": rec["risk"]["level"],
                              "Risk agrees": "yes" if r["risk"] == rec["risk"]["level"] else "NO",
                              "CRA issues": names(r["issues"]), "SCOPE issues": names(got),
                              "Issues agree": "yes" if set(r["issues"]) == got else "partly"})
            df = pd.DataFrame(table)
            n = len(df)
            high = df[df["CRA risk"] == "high"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk level agrees", f"{int((df['Risk agrees'] == 'yes').sum())} / {n}")
            c2.metric("High-risk visits caught", f"{int((high['SCOPE risk'] == 'high').sum())} / {len(high)}"
                      if len(high) else "none in set")
            c3.metric("False alarms (flagged high, CRA said lower)",
                      int(((df["SCOPE risk"] == "high") & (df["CRA risk"] != "high")).sum()))
            st.dataframe(df, hide_index=True, width="stretch")
            for r, rec in zip(rows, recs):
                if r["risk"] == rec["risk"]["level"]:
                    continue
                with st.expander(f"{r['id']}: CRA said {r['risk']}, SCOPE said {rec['risk']['level']}"):
                    st.text(r["text"])
                    st.dataframe(findings_table(rec, "active"), hide_index=True, width="stretch")
                    st.caption("Who is right? If the label looks wrong to you, tell us; if SCOPE is wrong, this "
                               "is what the next prompt or rubric change should fix.")
