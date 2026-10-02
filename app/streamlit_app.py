"""SCOPE Streamlit front end.

    streamlit run app/streamlit_app.py
    SCOPE_MODEL_DIR=models/scope-bert streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scope.data.generate import read_jsonl  # noqa: E402
from scope.data.handwritten import load_handwritten  # noqa: E402
from scope.predict import RuleBasedParser, load_parser  # noqa: E402
from scope.record import audit_summary, build_record, to_row  # noqa: E402
from scope.schema import ISSUE_BY_CODE, ISSUE_GROUPS  # noqa: E402

MODEL_DIR = os.environ.get("SCOPE_MODEL_DIR", "models/scope-bert")
# Public Hugging Face repo with the trained checkpoint; downloaded on first run if MODEL_DIR is empty.
HF_MODEL_REPO = os.environ.get("SCOPE_HF_MODEL", "FabcanK6/scope-bert")
CORPUS = ROOT / "data" / "corpus.jsonl"
SPAN_COLORS = {"VISIT_TYPE": "#4C78A8", "VISIT_DATE": "#B279A2", "SITE": "#F58518", "MONITOR": "#72B7B2",
               "PI": "#54A24B", "SCREENED": "#9D755D", "ENROLLED": "#9D755D", "ACTION": "#E45756",
               "OWNER": "#EECA3B", "DUE": "#FF9DA6"}
RISK_COLORS = {"low": "#2E7D32", "medium": "#ED6C02", "high": "#C62828"}

st.set_page_config(page_title="SCOPE - Site Visit Note Intelligence", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Downloading the SCOPE model (first run only)...")
def ensure_model() -> bool:
    if (Path(MODEL_DIR) / "scope_model.pt").exists():
        return True
    if not HF_MODEL_REPO:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(HF_MODEL_REPO, local_dir=MODEL_DIR)
    except Exception as exc:  # network issues, missing repo, ...
        st.sidebar.warning(f"Could not download `{HF_MODEL_REPO}`: {exc}")
    return (Path(MODEL_DIR) / "scope_model.pt").exists()


@st.cache_resource
def get_parser(backend: str):
    return RuleBasedParser() if backend == "rules" else load_parser(MODEL_DIR, backend=backend)


@st.cache_resource(show_spinner="Building the similar-visit index (first run only)...")
def get_index():
    from scope.search import NoteIndex

    rows = read_jsonl(CORPUS)
    try:
        return NoteIndex(rows, backend="embeddings")
    except Exception:  # sentence-transformers unavailable -> keyword search
        return NoteIndex(rows, backend="tfidf")


@st.cache_data
def examples() -> dict[str, str]:
    notes = {r["id"]: r["text"] for r in load_handwritten()}
    picks = {"Quick field note (high risk)": "hw-01", "Structured report": "hw-03", "Clean remote visit": "hw-02",
             "Missed SAE": "hw-04", "E-mail style": "hw-18", "Bullet notes": "hw-16", "Drug and dosing": "hw-07"}
    return {label: notes[i] for label, i in picks.items()}


def highlight(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["char_start"]):
        if sp["char_start"] < pos:
            continue
        out.append(html.escape(text[pos:sp["char_start"]]))
        color = SPAN_COLORS.get(sp["label"], "#999")
        out.append(f"<span style='background:{color}33;border-bottom:2px solid {color};padding:1px 3px;"
                   f"border-radius:4px'>{html.escape(text[sp['char_start']:sp['char_end']])}"
                   f"<sub style='color:{color};font-size:0.65em;margin-left:3px'>{sp['label']}</sub></span>")
        pos = sp["char_end"]
    out.append(html.escape(text[pos:]))
    return "<div style='line-height:2.0;font-size:0.95rem'>" + "".join(out).replace("\n", "<br>") + "</div>"


def risk_badge(level: str, conf: float | None) -> str:
    c = RISK_COLORS[level]
    extra = f" &middot; {conf:.0%} confidence" if conf else ""
    return (f"<span style='background:{c};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>"
            f"{level.upper()} RISK</span><span style='color:gray'>{extra}</span>")


# ---------------------------------------------------------------------------
st.title("🩺 SCOPE")
st.caption("Site Communication & Oversight Processing Engine · turns free-text site-visit notes into "
           "structured, audit-ready visit records")

with st.sidebar:
    has_model = ensure_model()
    backend = st.radio("Parser", ["hybrid", "bert", "rules"], index=0 if has_model else 2,
                       help="'hybrid' (recommended): BERT for risk and issues, rules for dates, site and counts. "
                            "'bert': the fine-tuned model alone. 'rules': the regex + keyword baseline.")
    if backend != "rules" and not has_model:
        st.warning(f"No checkpoint at `{MODEL_DIR}`, so the rule-based parser is used instead.")
    st.markdown("**Example notes**")
    for label, text in examples().items():
        if st.button(label, width="stretch"):
            st.session_state["note"] = text
    st.caption("All notes, sites and people in this app are fictional.")

tab_one, tab_batch = st.tabs(["Analyze a note", "Portfolio view"])

with tab_one:
    note = st.text_area("Site-visit note", key="note", height=220,
                        placeholder="Paste a monitoring visit note, field note or visit e-mail...")
    if note.strip():
        parser = get_parser(backend)
        rec = parser.analyze(note)
        v = rec["visit"]

        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        c1.markdown(risk_badge(rec["risk"]["level"], rec["risk"]["confidence"]), unsafe_allow_html=True)
        c2.metric("Site", v["site"]["id"] or "?")
        c3.metric("Visit date", v["visit_date"]["iso"] or (v["visit_date"]["text"] or "?"))
        c4.metric("Visit type", v["visit_type"]["code"] or "?")
        for w in rec["warnings"]:
            st.warning(w)

        t_rec, t_note, t_sim, t_audit, t_json = st.tabs(
            ["Visit record", "Highlighted note", "Similar past visits", "Audit summary", "JSON"])
        with t_rec:
            left, right = st.columns(2)
            with left:
                st.subheader("Active issues")
                if rec["issues"]:
                    for grp in ISSUE_GROUPS:
                        items = [i for i in rec["issues"] if i["group"] == grp]
                        if items:
                            st.markdown(f"**{grp}**")
                            for i in items:
                                conf = f" ({i['confidence']:.0%})" if i["confidence"] < 1 else ""
                                st.markdown(f"- {i['display']}{conf}")
                else:
                    st.success("No active issues found.")
                if rec["risk"]["probabilities"]:
                    probs = pd.DataFrame({"risk": list(rec["risk"]["probabilities"]),
                                          "probability": list(rec["risk"]["probabilities"].values())})
                    st.altair_chart(alt.Chart(probs).mark_bar().encode(
                        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("risk:N", sort=["low", "medium", "high"]),
                        color=alt.Color("risk:N", scale=alt.Scale(domain=list(RISK_COLORS),
                                                                  range=list(RISK_COLORS.values())), legend=None),
                    ).properties(height=120, title="Risk probabilities"), width="stretch")
                    st.caption("Calibrated on synthetic validation notes; on differently written notes the model "
                               "can be more confident than it should be.")
            with right:
                st.subheader("Visit details")
                st.markdown(
                    f"- **Visit type:** {v['visit_type']['name'] or '-'}  \n"
                    f"- **Monitor:** {v['monitor'] or '-'}  \n- **PI:** {v['pi'] or '-'}  \n"
                    f"- **Screened / enrolled:** {v['screened'] if v['screened'] is not None else '-'} / "
                    f"{v['enrolled'] if v['enrolled'] is not None else '-'}")
                st.subheader("Action items")
                if rec["actions"]:
                    st.dataframe(pd.DataFrame([{"Action": a["action"], "Owner": a["owner"] or "-",
                                                "Due": a["due_date"] or a["due"] or "-"} for a in rec["actions"]]),
                                 hide_index=True, width="stretch")
                else:
                    st.info("No action items found.")
        with t_note:
            st.markdown(highlight(note, rec["spans"]), unsafe_allow_html=True)
        with t_sim:
            only_shared = st.checkbox("Only show visits that share an active issue", value=bool(rec["issues"]))
            index = get_index()
            hits = index.search(note, k=5, issue_filter=[i["code"] for i in rec["issues"]] if only_shared else None)
            st.caption(f"Search backend: {index.backend} over {len(index.rows)} past (synthetic) visits.")
            for h in hits:
                m = h["meta"]
                issues = ", ".join(ISSUE_BY_CODE[c].display for c in h["issues"]) or "no active issues"
                with st.expander(f"{h['score']:.2f} · Site {m.get('site_id', '?')} · {m.get('visit_date', '?')} · "
                                 f"{h['risk']} risk · {issues}"):
                    st.text(h["text"])
        with t_audit:
            md = audit_summary(rec)
            st.markdown(md)
            st.download_button("Download summary (.md)", md, "scope_visit_summary.md", "text/markdown")
        with t_json:
            st.json(rec)
            st.download_button("Download record (.json)", json.dumps(rec, indent=2), "scope_visit_record.json",
                               "application/json")

with tab_batch:
    st.markdown("Run SCOPE over many visits at once and see where the risk is. Upload a CSV with a `note` "
                "column, or use the bundled sample of past (synthetic) visits.")
    up = st.file_uploader("CSV with a 'note' column", type=["csv"])
    n = st.slider("Sample size", 10, 100, 40, step=10)
    texts: list[str] = []
    if up is not None:
        df_in = pd.read_csv(up)
        if "note" not in df_in.columns:
            st.error("The CSV needs a column named 'note'.")
        else:
            texts = df_in["note"].astype(str).tolist()[:500]
    elif st.button("Analyze sample visits"):
        texts = [r["text"] for r in read_jsonl(CORPUS)[:n]]
    if texts:
        parser = get_parser(backend)
        with st.spinner(f"Analyzing {len(texts)} notes..."):
            preds = parser.predict_batch(texts)
            recs = [build_record(t, p) for t, p in zip(texts, preds)]
        df = pd.DataFrame([to_row(r) for r in recs])
        c1, c2, c3 = st.columns(3)
        c1.metric("Visits", len(df))
        c2.metric("High risk", int((df["risk"] == "high").sum()))
        c3.metric("Open action items", int(df["n_actions"].sum()))
        counts = pd.DataFrame([{"issue": ISSUE_BY_CODE[i["code"]].display, "group": i["group"]}
                               for r in recs for i in r["issues"]])
        if not counts.empty:
            st.altair_chart(alt.Chart(counts).mark_bar().encode(
                x=alt.X("count():Q", title="Visits with this issue"),
                y=alt.Y("issue:N", sort="-x", title=None), color=alt.Color("group:N", title="Group"),
            ).properties(height=320, title="Active issues across visits"), width="stretch")
        order = {"high": 0, "medium": 1, "low": 2}
        st.dataframe(df.sort_values("risk", key=lambda s: s.map(order)), hide_index=True, width="stretch")
        st.download_button("Download table (.csv)", df.to_csv(index=False), "scope_portfolio.csv", "text/csv")
