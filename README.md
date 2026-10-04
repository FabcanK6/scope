# SCOPE: Site Communication & Oversight Processing Engine

[![tests](https://github.com/FabcanK6/scope/actions/workflows/tests.yml/badge.svg)](https://github.com/FabcanK6/scope/actions/workflows/tests.yml)

SCOPE reads free-text clinical trial monitoring visit notes (formal visit reports, quick field notes, visit e-mails) and turns each one into a record anyone working on the study can act on:

- **Risk level** (high / medium / low) computed in code by a severity rubric, with the reason ("1 critical finding")
- **Every finding with its evidence**: what is an active problem, what was fixed during the visit, and what was checked and fine, each with the sentence from the note that shows it
- **Judged by the study's own protocol**: upload a protocol once and SCOPE drafts the study's rules (SAE definitions, reporting deadlines, visit windows, important deviations), each quoted with its page; deadlines are counted by code, not guessed
- **Visit details, open action items** with owner and due date, **a full visit report draft** from rough notes, and **a draft follow-up letter** to the investigator
- **Site history across visits**: each visit is added to its site, so the next visit opens with what is still open, and a problem that was active last time is pointed out
- **Learns from the people who use it**: corrections and ratings that three users agree on change how SCOPE reads similar notes, for everyone, without a new release
- **Any AI model**: the free shared Gemini engine, or your own key for Gemini, OpenAI, Anthropic Claude or any OpenAI-compatible service
- **Portfolio view** across many visits, and similar past visits

**Live app:** [scope-fabcank6.streamlit.app](https://scope-fabcank6.streamlit.app) · try a demo study in two clicks, or rate a practice visit in **Help SCOPE learn**.

All notes, sites, studies and people in this repository and the app are fictional. Only use fictional or de-identified notes.

---

## The problem

Monitoring visit notes hold the earliest warning signs at a trial site: a consent signed after procedures, a hospitalization nobody reported as an SAE, a dosing error. But they are long free text, and most of what they say is "checked, no problem". Someone has to read every note to find the few lines that matter, track the follow-ups, and show later that each finding was handled.

SCOPE does the first read. It is built to earn its users' trust rather than to impress: every risk call comes with the findings that caused it, every finding comes with a quote from the note, and anything that cannot be traced back to the note is thrown away.

## How it works

```text
protocol ─► study profile: rubric + the study's rules (each quoted with its page) + reporting deadlines
note ─► AI model (Gemini, or your own provider): instructions + the study profile + worked examples
          + rulings the community agreed on for similar notes
          returns JSON: visit details, every finding (topic, status, rule, severity, quoted evidence), action items
     ─► verification (plain code): every quote, name, date and action must appear in the note, or it is dropped
     ─► deadlines (plain code): elapsed calendar days, business days or hours, from the note's dates
     ─► rubric (plain code): protocol-rule severities, escalation, risk = high / medium / low
     ─► safety net (plain code): alerts when the note signals an SAE, consent, dosing or IRB problem the reading missed
     ─► visit record · highlighted note · audit summary · follow-up letter · portfolio table
```

- **The LLM reads; code decides.** The model's job is reading comprehension: is this sentence a problem, something fixed on site, or a confirmation that all is well? The risk level is never the model's opinion. It is computed from the verified findings with a fixed rubric, so the same findings always give the same answer and the rubric lives in an editable study profile (`scope/profile.py`).
- **Nothing without evidence.** Every finding must quote the note. SCOPE checks each quote (ignoring case and spacing) and drops findings whose quote is not there, as well as names, dates and action items that do not appear in the note. Anything dropped is listed so the reviewer can see it.
- **Fails loudly, not quietly.** An empty or garbled LLM answer is retried and never scored (an empty answer would otherwise look like a clean, low-risk visit). A plain-code safety net also scans the note for words that signal a possible SAE, consent problem, dosing error or IRB lapse; if the LLM reported nothing on that topic, SCOPE shows a red safety alert. The LLM is called with Gemini's default temperature, as Google recommends for Gemini 3 models (a low temperature can cause looping and degraded answers).
- **Few-shot, not fine-tuned.** The prompt contains the rubric and three worked examples in three note styles (a formal report with problems fixed on site and one open major issue, bullet notes with a critical finding, and a short e-mail with only a minor issue). The examples were written for the prompt and are not in any test set or in the app's example notes, so the demo and the evaluation are not answered in advance (a unit test checks this). SCOPE improves through its study profiles and the rulings its users agree on (below), not by retraining the AI model. The app's **Accuracy check** tab runs SCOPE on labelled notes and shows where it agrees and disagrees with the labels.
- **Free and light.** It runs on the free tier of the Gemini API through Python's standard library; the app has no deep-learning framework to install. Answers are cached for 24 hours so repeated notes cost nothing, each visitor is capped at 25 new requests on the shared engine, and models whose free daily quota is used up are skipped until it resets.

### Why an LLM and not the fine-tuned model

The first version used a fine-tuned BERT model (results below). It was near-perfect on generated notes and caught 7 of 8 high-risk hand-written notes, but a real-style formal report broke it: a routine visit where consent, SDV, drug accountability and storage were all confirmed fine, and one data-entry gap was fixed on site, came back as **high risk at 100% confidence**. The model had learned that mentioning a topic usually means a problem. Real reports mention every topic, mostly to say it is fine, and judging that is reading comprehension, which is what large language models do well. The fine-tuned model and its training pipeline stay in the repository as the baseline.

### Protocol first: build a study profile from the protocol

Protocols disagree on the things that decide risk: one defines an event as an SAE and wants it reported within 2 business days, another allows 3 calendar days, a third says the same event is only an AE (or that disease progression or a pre-planned hospitalization is not an SAE at all). So SCOPE reads the protocol first:

1. Upload the protocol (PDF, Word or text) in the Study setup tab and choose a starting preset.
2. The LLM drafts the study-specific rules a monitor needs: SAE definitions, exceptions and reporting deadlines, other expedited reporting, visit windows, key assessments, eligibility, dosing hold and stop rules, storage and excursion handling, consent, and the protocol's list of important deviations. Long protocols are cut to the pages that matter for monitoring.
3. SCOPE checks every rule's quote against the protocol text and finds its page; rules whose quote is not in the protocol are dropped.
4. The study lead rewords, re-grades or rejects each rule, then creates the study profile. Accepted rules carry their protocol citation (number, version, page), and the change is logged.
5. Every visit for that study is then judged against its own protocol: the LLM is told that study rules override general practice and to work out elapsed time (calendar or business days) from the dates in the note.
6. **Deadlines are counted by code, not by the AI.** Language models are unreliable at calendar arithmetic, so for topics with a reporting deadline (24 hours by default; e.g. 2 business days or 3 calendar days under a protocol) the model only copies the start date (e.g. site awareness) and the report date from the note. SCOPE checks both dates are in the note, counts calendar days, business days (Monday to Friday) or hours, and decides on time or late itself, overriding the model and saying so. With dates only, an hours deadline reported the next day is flagged as unclear rather than guessed.
7. **The protocol sets the severity.** Study rules are numbered for the model. When a finding falls under a rule that states a severity ("a tumour assessment outside its window is an important deviation [major if broken]"), the model names the rule and SCOPE applies that rule's severity itself, so a model that under- or over-rates the finding is overruled by the protocol. The findings table shows which rule set the severity and its page.

Five fictional demo studies ship with the app (see below) to try this with. Free-tier LLM requests may be used by the provider, so only use public (e.g. ClinicalTrials.gov) or fictional protocols until an enterprise LLM agreement is in place.

### Try it in two clicks: demo studies

Pick a study in the sidebar, then a note. Five fictional studies ship with SCOPE, each with a short protocol PDF, a ready-made study profile whose rules quote the protocol (with page numbers), and three example notes with the verdict a reviewer would expect:

| Study | Area | What it shows |
|---|---|---|
| ZLV-301 | Dermatology (oral tablets) | SAEs within 2 business days; elective pre-planned hospitalizations are not SAEs; a Week 16 visit 6 days late is inside its ±7-day window |
| ONC-210 | Oncology (IV infusion) | Hospitalization for disease progression is not an SAE; dosing with low neutrophils is critical; scans every 6 weeks ±7 days |
| VAX-118 | Vaccines | Cold chain: doses given after an unreported fridge excursion; the Day 29 immunogenicity sample is key |
| CRD-07 | Medical device | SAEs and device deficiencies within 3 calendar days (the default 24 hours would call the same report late); only trained implanters |
| PED-44 | Pediatrics | Parent consent plus child assent from age 7; weight-based dosing; status epilepticus is always an SAE |

The Accuracy check runs all 15 demo notes, each under its own protocol. `scripts/make_demo_studies.py` rebuilds the demos and fails if any rule's quote is not in its protocol.

### Your studies are remembered

Set a study up once. A study profile built from your protocol, edited, or loaded from a file is saved automatically under **My studies** (★) in your browser, together with any corrections you approve. Next week or next quarter, pick it from the study list and score the new visit: no need to upload the protocol again. Nothing is stored on the server; download a copy to back it up or use it on another computer.

### From rough notes to a visit report

The **Visit report** tab turns a rough note (shorthand, bullets, a dictated run-on) into a structured monitoring visit report: visit details, summary, findings requiring action, items resolved during the visit, areas reviewed with no issues, an action-item table and the next visit. It is written only from the note and SCOPE's verified reading (findings with their quotes, actions, visit details); anything the note does not say is left as a `[placeholder]`, and the app lists the placeholders and any missing section. Edit it in place and download it as Markdown (`scope/reports.py`).

### Site history and open actions across visits

Every visit read for a study is added to its site's history: date, visit type, risk, active findings and action items (`scope/tracker.py`). At the next visit to the same site, SCOPE shows **Still open from earlier visits** with a tick box per item, and points out a problem that was also active at the previous visit (the rubric raises repeat findings only when the note says so, so SCOPE leaves that call to the reader). The **Sites & actions** tab lists every site in the study with its visits, last risk and open actions, where items can be ticked done. Like saved studies, the history stays in the user's own browser.

### SCOPE learns from the people who use it

SCOPE does not wait for a new release to get better. Under every result there are two ways to teach it:

- **SCOPE got this right**: one click shares the note and SCOPE's reading.
- **Disagree with SCOPE? Correct it**: say what the finding should be, with the words from the note and one line on why, and tick **Share it**.

**The community reviews, not one person.** Shared cases appear in the **Help SCOPE learn** tab, where other users agree or disagree. A case goes live once 3 different people agree (the person who shared it counts as one) and at least 75% of votes agree; it is dropped when as many disagree. From then on SCOPE shows an agreed correction to the AI model as a ruling whenever it reads a similar note, for every user. The same tab has **Rate a practice visit**: fictional notes that anyone can rate low, medium or high in two clicks; a note's label counts once 3 people agree and nobody rated it two levels away. Ratings are anonymous: each browser gets a random id, stored only as a short hash, so one browser votes once per item. A curator can still approve, reject or retire anything (Curator tools, behind `SCOPE_CURATOR_KEY`). Every ruling keeps who agreed, when and why. In the Accuracy check a note never sees a ruling made on that same note, so the score stays honest, and agreed practice notes become their own Accuracy check set.

Agreed practice notes and approved cases also train **SCOPE's own model**: a small text model (TF-IDF and logistic regression) trained on the expert-labelled notes, the practice notes the community agreed on, and every approved case. It retrains by itself in a few seconds whenever an approved case is added or retired, then measures itself on expert-labelled and community-agreed notes it was not trained on (5-fold, repeated 3 times; shared copies of a test note are kept out). It **switches itself on only when it clears a bar**: 80% agreement with the experts' risk levels and 80% of high-risk visits caught (`SCOPE_OWN_MODEL_BAR` changes the first). Once on, it gives a second opinion on every visit, raising a **Second look** warning when it reads a visit as high risk and the AI reading says low, and it gives a rough risk estimate when the AI model is unavailable (for example when the free quota runs out). Today, with 56 expert notes, it agrees about 64% of the time and catches about 82% of high-risk visits, so it stays in the background; the **Help SCOPE learn** tab shows its current score, so its progress is visible. The curator view exports the approved cases (`scope_training.jsonl`).

Only fictional or de-identified notes may be shared. The shared library is a private Hugging Face dataset; set it up in the app's secrets:

```toml
HF_TOKEN = "hf_..."                          # fine-grained token with write access to that one dataset only
SCOPE_LEARNING_REPO = "your-name/scope-learning"
SCOPE_CURATOR_KEY = "a passphrase only curators know"
```

Without these, SCOPE works as before and corrections improve each study only. For a local run, `SCOPE_LEARNING_DIR=/some/folder` keeps the library in a folder instead.

### Severity rubric (v3.3)

Version 3.2 was written and approved by an experienced clinical research professional; v3.3 adds two rulings: one sentence that shows two different problems counts under each topic, and a PI on leave with no covering investigator is one problem (PI oversight), not also a delegation problem. Each active finding scores 1 (minor), 3 (major) or 6 (critical), counting the worst finding per topic: any critical finding or two major findings make the visit **high** risk, one major finding or three minor findings make it **medium**, anything less is **low**. Findings corrected and verified during the visit do not count, and critical findings stay active even when a CAPA is in place.

Two escalation rules weigh a problem the way an experienced reviewer does. A **repeat finding** (also found at an earlier visit, or an earlier action still open) is raised one level. A problem affecting **3 or more subjects**, or described as site-wide, is raised one level, up to major. Both can apply, so a minor gap that affects five subjects and was cited last visit becomes critical. The LLM only reports the facts (repeat, number of subjects, and a quote showing it); SCOPE applies the rules, and only when the quote is really in the note.

### Issue types (22)

| Group | Issues |
|---|---|
| Patient safety & consent | late or missing SAE reporting, adverse event recording, informed consent, eligibility, safety reports to IRB and PI, blinding |
| Protocol & drug | protocol deviation, dosing error, IP accountability, IP storage and temperature, lab samples and kits |
| Data quality | data entry backlog, open or aging queries, SDV and source access, source documentation |
| Site operations | staff, training and delegation, PI oversight, enrollment, regulatory and essential documents, facility and equipment, follow-up of prior findings, site engagement |

Each issue type has minor, major and critical examples in the default study profile (`scope/profile.py`), editable per study in the app. The v1 baseline model keeps its original 12 issue types.


### Study profiles: the rubric is a feature, not code

Monitoring standards change with the protocol, the study and the sponsor, so the rubric lives in a **study profile** that the study lead can edit in the app (Study setup tab):

- **Topics and severities**: change what counts as minor, major or critical, switch off topics that do not apply, or add study-specific topics.
- **Study rules**: plain-language rules from the protocol ("A missed Cycle 1 Day 1 PK sample is critical"). They override the default rubric.
- **Escalation and thresholds**: how many subjects make a problem widespread, how far that raises it, whether repeats escalate, and the points for medium and high risk.
- **Learning from corrections**: any user can correct a result ("this should be minor, because..."). Once the study lead approves a correction, SCOPE shows it to the LLM as an example whenever it reads a similar note, so it adapts to the study without retraining. Notes with corrections can be re-checked in the Accuracy check.

Every change is versioned and logged, and every result records the profile name, version and fingerprint that scored it, so a QA reviewer can see exactly which rules were applied. Profiles are kept in the user's browser under **My studies** and can be downloaded as JSON files to back up, share or move to another computer. The default profile, **SCOPE standard**, is rubric v3.3 below.

## Setup (free Gemini API key)

1. Create a free key at [aistudio.google.com](https://aistudio.google.com) (Get API key → Create API key).
2. Deployed app: in Streamlit Community Cloud open the app's **Settings → Secrets** and add `GEMINI_API_KEY = "..."`. Never commit the key. Optionally pin a model with `GEMINI_MODEL = "..."`; by default SCOPE picks the newest available Gemini Flash model. If a model is not available on the free tier, is overloaded (HTTP 503), too slow (no answer within 60 seconds), or has used up its own free quota (HTTP 429; free quotas are per model and reset at midnight Pacific time), SCOPE moves on to the next model, so one busy or exhausted model does not stop the app. Gemini 3 models are asked to think at a low level (`GEMINI_THINKING`, default `low`) so a reading takes seconds. Under every result, **How this was read** lists each model tried and what happened.
3. Locally: `export GEMINI_API_KEY=...`, then `streamlit run app/streamlit_app.py` or `python -m scope.llm --file note.txt --letter`.

**Bring your own key, any provider.** In the sidebar, visitors can keep the app's free Gemini engine or bring their own key for Google Gemini, OpenAI, Anthropic Claude, or any OpenAI-compatible service (Azure OpenAI, Mistral, Groq, OpenRouter, a local Ollama or vLLM server). SCOPE lists the provider's models to choose from. Keys stay in the browser session and are sent only to that provider; own keys are not capped. Every provider's answer goes through the same schema, quote verification, rubric scoring and safety net, and each result says which provider and model read it. Answers are constrained to SCOPE's schema with each provider's structured output feature (Gemini `responseSchema`, OpenAI strict `json_schema`, Anthropic `output_config`), with a plain-JSON fallback for services that lack it. Only send fictional or de-identified notes; real study data needs an enterprise agreement with the provider.

## Evaluation

```bash
GEMINI_API_KEY=... python -m scope.evaluate --backend llm --data handwritten realistic --sleep 5
```

**Stress test** (25 notes): written to cover many writing styles, every visit type, all 22 issue types, both escalation rules and common traps (problems fixed on site, "no SAEs" negations, resolved past items, dates near the visit date). Labels were reviewed by an experienced clinical research professional, and none of these notes are in SCOPE's instructions, so this is the honest measure. **Practice notes** (25 more, `scope/data/practice.txt`) are labelled by the community in the app; each one joins the Accuracy check once three people agree on it.

### Results so far (LLM engine)

| Test set | Notes | Risk level agrees | High-risk visits caught | When |
|---|---|---|---|---|
| Stress test | 25 | 24 (96%) | 10 of 10 | Oct 2026, re-run after the quota reset (earlier: 23, 9 of 10); one false alarm, st-024 |
| Formal visit reports | 7 | 7 | all | Oct 2026, after rubric rulings |
| Demo studies (each under its own protocol) | 15 | 14 (93%) | 6 of 7 | Oct 2026, re-run after the fixes (earlier: 12 of 15, mostly the smallest model); the miss, vax-118-3, led to the "one sentence, two problems" rule |

Results vary with the model that answered: the free tier falls back to smaller models when the larger ones' daily quota is used up. Every result in the app says which model read it, and the Accuracy check can be re-run at any time.

Test sets: **realistic** (7 long, formal notes in the style of real monitoring reports, provided by an experienced clinical research professional) and **handwritten** (24 notes in other styles: field notes, e-mails, run-on sentences). The realistic notes shaped the rubric, so they are a development set; an independent set of notes that neither the prompt nor the code has seen is the next step.

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
  profile.py           study profiles: editable rubric, study rules, escalation, user corrections
  protocol.py          protocol intake: read a protocol, draft cited study rules, build a profile
  deadlines.py         reporting deadlines counted from the note's dates (calendar, business days, hours)
  learning.py          shared learning: shared cases, curator review, rulings used on similar notes
  ownmodel.py          SCOPE's own model: retrains on approved cases, switches itself on at a measured bar
  reports.py           visit report draft from rough notes and the verified reading
  tracker.py           site history and open action items across visits
  demos.py             the five fictional demo studies (profiles/demos/)
  providers.py         LLM providers: Gemini, OpenAI, Anthropic Claude, OpenAI-compatible (bring your own key)
  schema.py            issue types, severity points, risk thresholds
  search.py            similar past visits (TF-IDF; embeddings optional)
  metrics.py           risk, issue, span, action-item and calibration metrics
  evaluate.py          evaluation CLI for every engine
  data/realistic.txt   7 realistic notes (development set)
  data/stress.txt      25 stress-test notes with reviewed labels
  data/practice.txt    25 practice notes the community labels in the app
  data/handwritten.*   24 hand-written evaluation notes and their loader
  data/generate.py     synthetic note generator (v1 training data)
  model.py, train.py   v1 fine-tuned BERT (baseline)
  rules.py             regex + keyword baseline
  predict.py           v1 parsers (BERT, rules, hybrid)
profiles/demos/        five fictional demo studies: protocol PDF, study profile, example notes
app/streamlit_app.py   web app
notebooks/             Colab notebook for the v1 baseline
scripts/               teacher-note generation with an LLM; Amazon Comprehend Medical comparison
tests/                 unit tests (the LLM is replaced by a fake, so tests need no key or network)
```

## Limitations

- Needs an internet connection and API quota. When the free quota runs out, the app says so; visitors can use their own free key.
- LLM output can vary between runs and between model versions. Verification and the fixed rubric limit the impact, and `GEMINI_MODEL` pins a version; re-run the evaluation whenever the model changes.
- The realistic test notes are few and shaped the prompt. Real use needs a larger, independent set of de-identified notes reviewed by experienced clinical research professionals.
- The default rubric started from one experienced reviewer; it changes as the community agrees on rulings, and each organization can adapt it in a study profile.
- Community learning needs people: rulings and practice labels count only once three people agree, and anonymous browser ids can be reset, so a curator can retire anything that should not stand.
- SCOPE supports human review; it does not replace it.

## License

MIT
