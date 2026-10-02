# SCOPE: Site Communication & Oversight Processing Engine

[![tests](https://github.com/FabcanK6/scope/actions/workflows/tests.yml/badge.svg)](https://github.com/FabcanK6/scope/actions/workflows/tests.yml)

SCOPE reads free-text clinical trial site-visit notes (quick field notes, full monitoring reports, visit e-mails) and turns each one into a structured, audit-ready visit record:

- **Site risk level** (low / medium / high) with a calibrated confidence
- **Active issues** across 12 types, ignoring things the note says are *not* a problem ("no new deviations", "excursion resolved")
- **Visit metadata**: visit type, date, site, monitor, PI, screened and enrolled counts
- **Follow-up action items** with owner and due date
- **Similar past visits** found by meaning, not keywords (sentence embeddings + FAISS)

**Live app:** [scope-fabcank6.streamlit.app](https://scope-fabcank6.streamlit.app) · **Model:** [huggingface.co/FabcanK6/scope-bert](https://huggingface.co/FabcanK6/scope-bert) · **Train it yourself:** [Open in Colab](https://colab.research.google.com/github/FabcanK6/scope/blob/main/notebooks/train_on_colab.ipynb)

All notes, sites and people in this repository are synthetic.

---

## The problem

Monitoring visit notes hold the earliest signal that a site is in trouble: a consent signed after procedures, a hospitalization nobody reported as an SAE, queries piling up after a coordinator leaves. But the notes are free text. Someone has to read every one to build the risk view, track the follow-ups, and prove later that each finding was handled.

SCOPE does the first pass. A monitor or study manager pastes a note (or a batch of notes) and gets back a record they can review, correct and file.

## Example

Input:

```text
IMV - Site 104 - 12-Mar-2026
CRA: J. Okafor | PI: Dr. Patel
14 screened, 9 randomized
- 23 queries open > 60 days
- No new protocol deviations noted.
- subject 104-007 underwent study procedures before signing the informed consent form.
CRC to close open queries by 26-Mar-2026.
Action: Site to document the consent deviation (due next visit).
```

Output (abridged):

```json
{
  "visit": {"visit_type": {"code": "IMV"}, "visit_date": {"iso": "2026-03-12"}, "site": {"id": "104"},
            "monitor": "J. Okafor", "pi": "Dr. Patel", "screened": 14, "enrolled": 9},
  "risk": {"level": "high"},
  "issues": [{"code": "QUERY_AGING"}, {"code": "CONSENT"}],
  "actions": [
    {"action": "close open queries", "owner": "CRC", "due": "26-Mar-2026", "due_date": "2026-03-26"},
    {"action": "document the consent deviation", "owner": "Site", "due": "next visit", "due_date": null}
  ],
  "warnings": []
}
```

"No new protocol deviations noted" mentions deviations but is not a finding, so it is not flagged.

## How it works

```text
note ─► word tokens ─► BERT encoder ──┬─► [CLS] ─► risk head ──────► low / medium / high  ─► temperature scaling
                                      ├─► [CLS] ─► issue head ─────► 12 sigmoid outputs (multi-label)
                                      └─► every token ─► tag head ─► BIO tags: VISIT_TYPE, VISIT_DATE, SITE,
                                                                     MONITOR, PI, SCREENED, ENROLLED, ACTION, OWNER, DUE
                                   │
                                   ▼
                    record builder (deterministic): normalizes dates, counts and visit types,
                    groups ACTION/OWNER/DUE into action items, adds warnings
                                   │
                                   ▼
            visit record (JSON) · audit summary (Markdown) · portfolio table (CSV) · similar past visits
```

- **One model, three jobs.** A single fine-tuned encoder with three heads is trained on the sum of three losses (cross-entropy for risk, binary cross-entropy for issues, token cross-entropy for tags). Any Hugging Face encoder works; the notebook compares `bert-base-uncased` with `emilyalsentzer/Bio_ClinicalBERT`, which was pre-trained on clinical notes.
- **Calibrated risk.** After training, a temperature is fitted on the validation set so that "90% confident" means right about 90% of the time.
- **Deterministic post-processing.** The model decides *what* the note says; plain code turns that into dates, numbers and action items, so every field in the record can be traced back to highlighted words in the note.
- **Similar-visit search.** Notes are embedded with `sentence-transformers/all-MiniLM-L6-v2` and indexed with FAISS (inner product on normalized vectors = cosine similarity). A TF-IDF index with the same interface is the keyword baseline.
- **Rule baseline.** Regular expressions, a keyword list per issue type, NegEx-style negation cues and a points table for risk. It needs no model, so it is both the yardstick and the fallback.
- **Hybrid parser (default).** Risk and issue flags from the fine-tuned model; visit metadata and action items from the rules, with the model filling in action items the rules miss. This split comes straight from the evaluation below.

### Issue types

| Group | Issues |
|---|---|
| Data & queries | data entry backlog, open/aging queries, source data verification behind |
| Patient safety & consent | late or missing SAE reporting, informed consent issue |
| Protocol & drug | protocol deviation, IP accountability, IP temperature excursion |
| Site operations | staff turnover / training gap, PI oversight gap, enrollment behind target, regulatory binder / essential documents |

Risk follows a fixed rubric: each active issue scores 1 (minor), 3 (major) or 6 (critical) points; 0–1 = low, 2–5 = medium, 6+ = high. The model learns to read severity from the wording ("a handful of pages" vs "no data entered for the last 8 visits").

## Data

There is no public corpus of monitoring visit notes, so SCOPE is trained on a synthetic generator (`scope/data/generate.py`) that writes notes in three styles (terse field notes, sectioned reports, narrative paragraphs) with labelled spans, issue flags and risk. Every note mixes real findings with **negated and resolved mentions** ("no consent issues identified", "the excursion from January was assessed and the kits released"), which is where keyword systems fail.

| Split | Notes | What it tests |
|---|---|---|
| train / val | 12,000 / 1,000 | |
| `test` | 1,500 | new notes, phrasings seen in training |
| `test_unseen` | 1,500 | held-out sentence variants and date formats, plus an e-mail note style never used in training |
| `handwritten` | 24 | notes written by hand in a different voice: abbreviations, run-on sentences, typos (`scope/data/handwritten.txt`) |

The hand-written set is small but it is the most honest test: it was not produced by the generator at all.

## Results

Four parsers on three test sets. The **hybrid** (BERT for risk and issues, rules for format-bound fields) is what the app uses.

**Hand-written notes** (24 notes, never produced by the generator: the hardest and most realistic test)

| Parser | Risk accuracy | High-risk visits caught | Issue F1 | Metadata F1 | Action-item F1 | Risk calibration error |
|---|---|---|---|---|---|---|
| Rules | 0.500 | 1 of 8 | 0.720 | 0.957 | 0.758 | n/a |
| BERT (bert-base-uncased) | 0.708 | 7 of 8 | 0.800 | 0.794 | 0.500 | 0.300 |
| Bio_ClinicalBERT | 0.708 | 7 of 8 | 0.825 | 0.776 | 0.442 | 0.319 |
| **Hybrid** | **0.708** | **7 of 8** | **0.800** | **0.957** | **0.753** | 0.300 |

**Unseen phrasings** (1,500 generated notes with held-out sentence variants, date formats and an e-mail style)

| Parser | Risk accuracy | High-risk recall | Issue F1 | Metadata F1 | Action-item F1 | False alarms on negated mentions |
|---|---|---|---|---|---|---|
| Rules | 0.857 | 0.785 | 0.976 | 0.938 | 0.923 | 2.4% |
| BERT | 0.855 | 0.829 | 0.946 | 0.632 | 0.753 | 7.2% |
| Bio_ClinicalBERT | 0.836 | 0.787 | 0.923 | 0.680 | 0.622 | 0.2% |
| **Hybrid** | **0.855** | **0.829** | **0.946** | **0.938** | **0.913** | 7.2% |

**Seen phrasings** (1,500 new generated notes): BERT 0.995 risk accuracy and 1.000 on issues, spans and action items; rules 0.890 / 0.952 / 0.984 / 0.987. Near-perfect scores here mostly show that the model learned the generator, which is why the two harder sets matter.

What the numbers say:

- **The model is better at judgement, the rules at format.** On hand-written notes BERT catches 7 of 8 high-risk visits against 1 of 8 for the rules, because it reads severity and context rather than matching keywords. But on date formats and layouts it never saw, its metadata F1 drops to 0.63 while a date regex does not care. The hybrid keeps the best of both: model-level risk and issue detection with rule-level metadata (0.957 on hand-written notes).
- **Clinical pre-training did not help.** Bio_ClinicalBERT (pre-trained on hospital notes) was no better than plain `bert-base-uncased` and slightly worse on unseen phrasings. Site-visit notes are operational language (queries, SDV, delegation logs), not patient narrative.
- **Confidence does not travel.** Temperature scaling (T = 1.35) brings calibration error on in-distribution notes to 0.003, but on hand-written notes it is 0.30: the model reports near-100% confidence on some wrong answers. Calibration fitted on synthetic validation data does not carry over to a new writing style, so confidence is shown in the app but should not be trusted as a probability on real notes without recalibrating on real data.
- **Typical remaining errors:** "zero open queries" read as an open-queries issue; severity under-called on a short note that describes a critical consent finding in casual words; a missing essential document mentioned in passing ("the updated 1572 is not in the binder") not flagged.

Training: 12,000 notes, 3 epochs, batch 16, learning rate 5e-5 (heads 1e-3), about 9 minutes per model on a free Colab T4. An action item only counts as correct when the action text, owner and due date all match.

### Similar-visit search

A hit counts as relevant when it shares an active issue type with the query note (243 queries from the unseen set, 400 past visits).

| Retrieval | Precision@5 | Same risk level@5 |
|---|---|---|
| TF-IDF (keywords) | 0.795 | 0.491 |
| Sentence embeddings (all-MiniLM-L6-v2) + FAISS | 0.799 | 0.473 |

A tie. The past visits and the queries come from the same generator and share most of their vocabulary, which is the easy case for keyword search; embeddings are expected to pull ahead when the wording differs ("fridge alarm" vs "temperature excursion"). `python -m scope.search --eval --queries handwritten` runs the same comparison with the hand-written notes as queries.

## Comparison with Amazon Comprehend Medical

`scripts/compare_comprehend_medical.py` sends the same synthetic notes to Amazon Comprehend Medical (`DetectEntitiesV2`) and compares what can be compared: dates and person names. Comprehend Medical is a general medical NLP service; it has no concept of a visit type, an issue flag, a risk level or an action item, which is the gap a domain-tuned model fills. The script prints a cost estimate and only calls AWS with `--yes`.

```bash
pip install boto3 && aws configure
python scripts/compare_comprehend_medical.py --data data/test_unseen.jsonl --max-notes 50          # estimate only
python scripts/compare_comprehend_medical.py --data data/test_unseen.jsonl --max-notes 50 --yes    # run
```

## Quick start

```bash
git clone https://github.com/FabcanK6/scope.git && cd scope
pip install -r requirements-dev.txt

python -m scope.data.generate --out data                 # synthetic notes
python -m scope.evaluate --backend rules                 # rule baseline on all test sets
python -m scope.cli --backend rules "IMV Site 104 12-Mar-2026. 23 queries open > 60 days. CRC to close queries by 26-Mar-2026."

# fine-tune (a GPU helps; the Colab notebook takes about 10-15 minutes on a free T4)
python -m scope.train --model bert-base-uncased --out models/scope-bert --fp16
python -m scope.evaluate --model models/scope-bert --backend hybrid   # or --backend bert

python -m scope.search --eval                            # embeddings vs TF-IDF retrieval
streamlit run app/streamlit_app.py                       # downloads FabcanK6/scope-bert on first run
pytest -q
```

## Repository layout

```text
scope/
  schema.py            labels: risk levels, 12 issue types, span types
  text.py              tokenization with character offsets, BIO helpers
  data/generate.py     synthetic note generator (styles, negations, held-out phrasings)
  data/templates.py    sentence banks and value pools
  data/handwritten.*   24 hand-written evaluation notes and their loader
  rules.py             regex + keyword + negation baseline
  model.py             BERT encoder with risk, issue and tag heads; temperature scaling
  train.py             fine-tuning loop
  predict.py           BERT and rule parsers with one interface
  record.py            visit record, action-item assembly, audit summary
  search.py            similar-visit search (sentence embeddings + FAISS, TF-IDF baseline)
  metrics.py           risk, issue, span, action-item and calibration metrics
  evaluate.py          evaluation CLI
  cli.py               analyze one note from the command line
app/streamlit_app.py   web app: single note, highlighted spans, similar visits, audit summary, portfolio view
notebooks/             Colab training notebook
scripts/               Amazon Comprehend Medical comparison
tests/                 unit tests (including a tiny randomly initialised BERT)
```

## Limitations

- Trained on synthetic notes. Real notes are messier and use site- and sponsor-specific vocabulary; the hand-written set shows the drop to expect, and real use needs a labelled sample of real (de-identified) notes for evaluation and fine-tuning.
- Risk is defined by a fixed points rubric. Organizations weight findings differently, so the rubric (and labels) should be adapted before use.
- Confidence scores are calibrated on synthetic data only; on differently written notes the model can be confidently wrong (see Results).
- Notes longer than the model's 512-token window are cut off; the record shows a warning when that happens.
- SCOPE supports human review; it does not replace it. Every field links back to the words it came from so a reviewer can check it quickly.

## License

MIT
