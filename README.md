# SCOPE: Site Communication & Oversight Processing Engine

[![tests](https://github.com/FabcanK6/scope/actions/workflows/tests.yml/badge.svg)](https://github.com/FabcanK6/scope/actions/workflows/tests.yml)

SCOPE reads free-text clinical trial monitoring visit notes (formal visit reports, quick field notes, visit e-mails) and turns each one into a record a CRA or study manager can act on:

- **Risk level** (high / medium / low) computed by a severity rubric reviewed by an experienced CRA, with the reason ("1 critical finding")
- **Every finding with its evidence**: what is an active problem, what was fixed during the visit, and what was checked and fine, each with the sentence from the note that shows it
- **Visit details**: visit type, date, site, monitor, PI, screened and enrolled counts
- **Open action items** with owner and due date
- **A draft follow-up letter** to the investigator, ready to edit
- **Portfolio view** across many visits, and similar past visits

**Live app:** [scope-fabcank6.streamlit.app](https://scope-fabcank6.streamlit.app)

All notes, sites and people in this repository are fictional.

---

## The problem

Monitoring visit notes hold the earliest warning signs at a trial site: a consent signed after procedures, a hospitalization nobody reported as an SAE, a dosing error. But they are long free text, and most of what they say is "checked, no problem". Someone has to read every note to find the few lines that matter, track the follow-ups, and show later that each finding was handled.

SCOPE does the first read. It is built to earn a CRA's trust rather than to impress: every risk call comes with the findings that caused it, every finding comes with a quote from the note, and anything that cannot be traced back to the note is thrown away.

## How it works

```text
note ─► LLM (Google Gemini): instructions + severity rubric + three worked examples
          returns JSON: visit details, every finding (topic, status, severity, quoted evidence), action items
     ─► verification (plain code): every quote, name, date and action must appear in the note, or it is dropped
     ─► rubric (plain code): risk = high / medium / low from the verified active findings
     ─► visit record · highlighted note · audit summary · follow-up letter · portfolio table
```

- **The LLM reads; code decides.** The model's job is reading comprehension: is this sentence a problem, something fixed on site, or a confirmation that all is well? The risk level is never the model's opinion. It is computed from the verified findings with a fixed rubric, so the same findings always give the same answer and the rubric can be changed in one place (`scope/llm.py`).
- **Nothing without evidence.** Every finding must quote the note. SCOPE checks each quote (ignoring case and spacing) and drops findings whose quote is not there, as well as names, dates and action items that do not appear in the note. Anything dropped is listed so the reviewer can see it.
- **Fails loudly, not quietly.** An empty or garbled LLM answer is retried and never scored (an empty answer would otherwise look like a clean, low-risk visit). A plain-code safety net also scans the note for words that signal a possible SAE, consent problem, dosing error or IRB lapse; if the LLM reported nothing on that topic, SCOPE shows a red safety alert. The LLM is called with Gemini's default temperature, as Google recommends for Gemini 3 models (a low temperature can cause looping and degraded answers).
- **Few-shot, not fine-tuned.** The prompt contains the rubric and three worked examples in three note styles (a formal report with problems fixed on site and one open major issue, bullet notes with a critical finding, and a short e-mail with only a minor issue). The examples were written for the prompt and are not in any test set or in the app's example notes, so the demo and the evaluation are not answered in advance (a unit test checks this). Improving SCOPE means improving the rubric and examples and re-running the evaluation, not retraining a model. The app's **Accuracy check** tab runs SCOPE on these labelled notes and shows where it agrees and disagrees with the CRA's labels.
- **Free and light.** It runs on the free tier of the Gemini API through Python's standard library; the app has no ML framework to install. Answers are cached for 24 hours so repeated notes cost nothing, and each visitor is capped at 25 new requests.

### Why an LLM and not the fine-tuned model

The first version used a fine-tuned BERT model (results below). It was near-perfect on generated notes and caught 7 of 8 high-risk hand-written notes, but a real-style formal report broke it: a routine visit where consent, SDV, drug accountability and storage were all confirmed fine, and one data-entry gap was fixed on site, came back as **high risk at 100% confidence**. The model had learned that mentioning a topic usually means a problem. Real reports mention every topic, mostly to say it is fine, and judging that is reading comprehension, which is what large language models do well. The fine-tuned model and its training pipeline stay in the repository as the baseline.

### Severity rubric

Each active finding scores 1 (minor), 3 (major) or 6 (critical): any critical finding or two major findings make the visit **high** risk, one major finding or three minor findings make it **medium**, anything less is **low**. Critical findings include a late or unreported SAE, procedures before consent, an ineligible subject dosed, dosing errors, expired or compromised IP being used, enrolling after IRB approval lapsed, untrained staff running visits, and refusal of source access; they stay active even when a CAPA is in place. Findings corrected and verified during the visit do not count.

### Issue types

| Group | Issues |
|---|---|
| Data & queries | data entry backlog, open/aging queries, source data verification behind |
| Patient safety & consent | late or missing SAE reporting, informed consent issue |
| Protocol & drug | protocol deviation, IP accountability, IP temperature excursion |
| Site operations | staff turnover / training gap, PI oversight gap, enrollment behind target, regulatory binder / essential documents |


## Setup (free Gemini API key)

1. Create a free key at [aistudio.google.com](https://aistudio.google.com) (Get API key → Create API key).
2. Deployed app: in Streamlit Community Cloud open the app's **Settings → Secrets** and add `GEMINI_API_KEY = "..."`. Never commit the key. Optionally pin a model with `GEMINI_MODEL = "..."`; by default SCOPE picks the newest available Gemini Flash model. If a model is not available on the free tier, is overloaded (HTTP 503), or has used up its own free quota (HTTP 429; free quotas are per model and reset at midnight Pacific time), SCOPE moves on to the next model, so one busy or exhausted model does not stop the app.
3. Locally: `export GEMINI_API_KEY=...`, then `streamlit run app/streamlit_app.py` or `python -m scope.llm --file note.txt --letter`.

Visitors can also paste their own key in the app's sidebar; it stays in their browser session. Free-tier requests may be used by the provider, so only send fictional or de-identified notes. A deployment on real study data would need an enterprise LLM agreement covering PHI.

## Evaluation

```bash
GEMINI_API_KEY=... python -m scope.evaluate --backend llm --data handwritten realistic --sleep 5
```

Test sets: **realistic** (7 long, formal notes in the style of real CRA reports, provided by an experienced CRA) and **handwritten** (24 notes in other styles: field notes, e-mails, run-on sentences). The realistic notes shaped the rubric, so they are a development set; an independent set of notes that neither the prompt nor the code has seen is the next step.

### v1 training data

There is no public corpus of monitoring visit notes, so the v1 model was trained on a synthetic generator (`scope/data/generate.py`) that writes notes in three styles (terse field notes, sectioned reports, narrative paragraphs) with labelled spans, issue flags and risk. Every note mixes real findings with **negated and resolved mentions** ("no consent issues identified", "the excursion from January was assessed and the kits released"), which is where keyword systems fail.

| Split | Notes | What it tests |
|---|---|---|
| train / val | 12,000 / 1,000 | |
| `test` | 1,500 | new notes, phrasings seen in training |
| `test_unseen` | 1,500 | held-out sentence variants and date formats, plus an e-mail note style never used in training |
| `handwritten` | 24 | notes written by hand in a different voice: abbreviations, run-on sentences, typos (`scope/data/handwritten.txt`) |

The hand-written set is small but it is the most honest test: it was not produced by the generator at all.


### v1 results (fine-tuned BERT, before the LLM engine)

These numbers were measured with the first severity rubric, before the formal-report notes and rubric v2 were added. Four parsers on three test sets.

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


#### Similar-visit search (v1)

A hit counts as relevant when it shares an active issue type with the query note. The bank holds 400 past (synthetic) visits.

| Queries | Retrieval | Precision@5 | Same risk level@5 |
|---|---|---|---|
| 243 unseen-phrasing notes | TF-IDF (keywords) | 0.795 | 0.491 |
| | Sentence embeddings (all-MiniLM-L6-v2) + FAISS | 0.799 | 0.473 |
| 19 hand-written notes | TF-IDF (keywords) | **0.821** | **0.516** |
| | Sentence embeddings (all-MiniLM-L6-v2) + FAISS | 0.758 | 0.368 |

I expected general-purpose embeddings to pull ahead when the wording changes. They did not: on hand-written queries, keyword search was better (with only 19 queries, the gap is about six hits). The likely reasons are that domain terms such as "ICF", "SAE" and "SDV" carry most of the signal and TF-IDF weights them directly, while an off-the-shelf sentence model also encodes writing style and treats "no SAEs reported" as close to "SAE reported late". What makes search useful in practice is the structured filter: the app can restrict results to past visits that share an issue the model detected, which is a precise signal rather than a fuzzy one. Next steps would be a domain-tuned embedding model or hybrid keyword + vector ranking, judged on real queries.


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
export GEMINI_API_KEY=...                                   # free key from aistudio.google.com

streamlit run app/streamlit_app.py
python -m scope.llm --file note.txt --letter                # one note from the command line
python -m scope.evaluate --backend llm --data handwritten realistic --sleep 5
pytest -q

# v1 baseline (fine-tuned BERT): see notebooks/train_on_colab.ipynb
pip install -r requirements-train.txt
python -m scope.data.generate --out data && python -m scope.evaluate --backend rules
```

## Repository layout

```text
scope/
  engine.py            the LLM engine: prompt with rubric and worked examples, verification, record
  llm.py               Gemini client (standard library), rubric scoring, quote checks, follow-up letter
  record.py            visit record, date and count normalization, audit summary
  schema.py            issue types, severity points, risk thresholds
  search.py            similar past visits (TF-IDF; embeddings optional)
  metrics.py           risk, issue, span, action-item and calibration metrics
  evaluate.py          evaluation CLI for every engine
  data/realistic.txt   7 realistic notes (development set)
  data/handwritten.*   24 hand-written evaluation notes and their loader
  data/generate.py     synthetic note generator (v1 training data)
  model.py, train.py   v1 fine-tuned BERT (baseline)
  rules.py             regex + keyword baseline
  predict.py           v1 parsers (BERT, rules, hybrid)
app/streamlit_app.py   web app
notebooks/             Colab notebook for the v1 baseline
scripts/               teacher-note generation with an LLM; Amazon Comprehend Medical comparison
tests/                 unit tests (the LLM is replaced by a fake, so tests need no key or network)
```

## Limitations

- Needs an internet connection and API quota. When the free quota runs out, the app says so; visitors can use their own free key.
- LLM output can vary between runs and between model versions. Verification and the fixed rubric limit the impact, and `GEMINI_MODEL` pins a version; re-run the evaluation whenever the model changes.
- The realistic test notes are few and shaped the prompt. Real use needs a larger, independent set of de-identified notes reviewed by CRAs.
- The rubric reflects one experienced reviewer; organizations weight findings differently and should adapt it.
- SCOPE supports human review; it does not replace it.

## License

MIT
