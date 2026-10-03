"""Synthetic site-visit notes with gold labels.

    python -m scope.data.generate --out data
    python -m scope.data.generate --out data --n-train 12000 --n-val 1000 --n-test 1500 --n-unseen 1500

Writes ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` (seen phrasings),
``test_unseen.jsonl`` (held-out phrasings, date formats and an e-mail note style
never used in training) and ``corpus.jsonl`` (a bank of past visits for search).

Every note is synthetic: sites, people and subjects are invented.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from scope.data import templates as T
from scope.schema import ISSUE_BY_CODE, V1_ISSUE_CODES, SEVERITY_POINTS, risk_from_points
from scope.text import char_spans_to_bio, tokenize

_SLOT_RE = re.compile(r"\[\[(\w+):(\w+)\]\]|\{(\w+)\}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(template: str, values: dict) -> tuple[str, list[tuple[str, int, int]]]:
    """Fill a template; return the text and its labelled character spans."""
    out, spans, pos = [], [], 0
    length = 0
    for m in _SLOT_RE.finditer(template):
        lit = template[pos:m.start()]
        out.append(lit)
        length += len(lit)
        label, slot = (m.group(1), m.group(2)) if m.group(1) else (None, m.group(3))
        val = str(values[slot])
        if label:
            spans.append((label, length, length + len(val)))
        out.append(val)
        length += len(val)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


class NoteBuilder:
    def __init__(self):
        self.text = ""
        self.spans: list[tuple[str, int, int]] = []

    def add(self, piece: str, spans: list[tuple[str, int, int]] | None = None, sep: str = "\n") -> None:
        if self.text:
            self.text += sep
        base = len(self.text)
        self.text += piece
        for label, s, e in spans or []:
            self.spans.append((label, base + s, base + e))


def format_date(d: date, fmt: str) -> str:
    s = d.strftime(fmt)
    if fmt == "%d%b%Y":
        s = s.upper()
    if fmt in ("%m/%d/%y",):  # 03/07/26 -> 3/7/26
        s = "/".join(p.lstrip("0") or "0" for p in s.split("/"))
    return s


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# ---------------------------------------------------------------------------
# Note sampler
# ---------------------------------------------------------------------------
class Sampler:
    def __init__(self, rng: random.Random, unseen: bool):
        self.rng = rng
        self.unseen = unseen

    def pick(self, items: list):
        seen, held = T.split_variants(items)
        pool = held if (self.unseen and held) else seen
        return self.rng.choice(pool)

    def number(self, n: int, words_ok: bool) -> str:
        if words_ok and n <= 12 and self.rng.random() < 0.3:
            return T.NUMBER_WORDS[n]
        return str(n)


def _sentence_values(rng: random.Random, ctx: dict) -> dict:
    return {
        "n_small": rng.randint(2, 6), "n": rng.randint(3, 12), "n_big": rng.randint(15, 80),
        "n_huge": rng.randint(100, 400), "days": rng.randint(10, 60), "days_small": rng.randint(2, 5),
        "pct_hi": rng.randint(80, 95), "pct_lo": rng.randint(20, 60),
        "subj": rng.choice([f"subject {ctx['site_id']}-{rng.randint(1, 40):03d}",
                            f"{ctx['site_id']}-{rng.randint(1, 40):03d}", f"Subj {rng.randint(1, 40):03d}"]),
        "ver": rng.randint(2, 6), "crit": rng.randint(1, 12), "visit_no": rng.randint(2, 12),
        "kit": rng.randint(10000, 99999), "temp": rng.randint(9, 14), "hour": rng.randint(7, 10),
        "subj2": f"{ctx['site_id']}-{rng.randint(1, 40):03d}",
        "enr_lag": ctx["ne"], "target": ctx["ne"] + rng.randint(5, 25),
        "other_date": format_date(ctx["visit_date"] - timedelta(days=rng.randint(20, 120)), ctx["date_fmt"]),
        "future_date": format_date(ctx["visit_date"] + timedelta(days=rng.randint(28, 90)), ctx["date_fmt"]),
    }


def sample_note(rng: random.Random, unseen: bool = False, style: str | None = None) -> dict:
    S = Sampler(rng, unseen)
    vt = rng.choices(list(T.VISIT_TYPE_WEIGHTS), weights=list(T.VISIT_TYPE_WEIGHTS.values()))[0]
    if style is None:
        if unseen:
            style = rng.choices(["field", "report", "narrative", "formal", "email"],
                                weights=[0.15, 0.15, 0.1, 0.2, 0.4])[0]
        else:
            style = rng.choices(["field", "report", "narrative", "formal"], weights=[0.3, 0.25, 0.15, 0.3])[0]
    formal = style == "formal"

    site_id = rng.choice([rng.randint(100, 999), rng.randint(1001, 3999)])
    visit_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 630))
    date_fmt = S.pick(T.FORMAL_DATE_FORMATS if formal else T.DATE_FORMATS)
    cra_first, cra_last = rng.choice(T.FIRST_NAMES), rng.choice(T.LAST_NAMES)
    pi_first, pi_last = rng.choice(T.FIRST_NAMES), rng.choice([n for n in T.LAST_NAMES if n != cra_last])
    ns = 0 if vt == "SIV" else rng.randint(1, 60)
    ne = 0 if vt == "SIV" else rng.randint(0, ns)
    ctx = {"site_id": site_id, "visit_date": visit_date, "date_fmt": date_fmt, "ne": ne}

    words_ok = style in ("report", "narrative", "email")
    values = {
        "vt": S.pick(T.VISIT_TYPE_SURFACES[vt]),
        "vt_title": T.VISIT_TYPE_TITLES[vt],
        "site": S.pick(T.SITE_FORMATS).format(id=site_id),
        "date": format_date(visit_date, date_fmt),
        "cra": rng.choice([f"{cra_first[0]}. {cra_last}", f"{cra_first} {cra_last}"]),
        "pi": rng.choice([f"Dr. {pi_last}", f"Dr. {pi_first} {pi_last}", f"{pi_first} {pi_last}"]),
        "ns": S.number(ns, words_ok), "ne": S.number(ne, words_ok),
    }

    # --- issues --------------------------------------------------------------
    candidates = list(T.VISIT_TYPE_ISSUES.get(vt, V1_ISSUE_CODES))
    if vt == "COV":
        candidates.remove("ENROLLMENT_LAG")
    if formal:  # real reports: mostly clean, many topics confirmed as fine
        k = rng.choices([0, 1, 2, 3, 4], weights=[0.3, 0.35, 0.2, 0.1, 0.05])[0]
    else:
        k = rng.choices([0, 1, 2, 3, 4, 5], weights=[0.2, 0.3, 0.25, 0.13, 0.08, 0.04])[0]
    active = rng.sample(candidates, min(k, len(candidates)))
    bank = T.FORMAL_ISSUE_SENTENCES if formal else T.ISSUE_SENTENCES
    sev_weights = {"minor": 0.45, "major": 0.4, "critical": 0.15}
    severities = {}
    for c in active:
        levels = [lv for lv in sev_weights if bank[c].get(lv)]
        severities[c] = rng.choices(levels, weights=[sev_weights[lv] for lv in levels])[0]
    rest = [c for c in candidates if c not in active]
    n_inactive = rng.randint(2, 5) if formal else rng.choices([0, 1, 2, 3], weights=[0.3, 0.35, 0.25, 0.1])[0]
    inactive_codes = rng.sample(rest, min(len(rest), n_inactive))
    inactive = {c: rng.choice(["negated", "negated", "resolved"]) for c in inactive_codes}
    risk = risk_from_points(sum(SEVERITY_POINTS[s] for s in severities.values()))

    mentions = []  # (group, text, spans, code)
    for code, kind in list(severities.items()) + list(inactive.items()):
        tmpl = S.pick(bank[code][kind])
        txt, sp = render(tmpl, _sentence_values(rng, ctx))
        mentions.append((ISSUE_BY_CODE[code].group, txt, sp, code))
    rng.shuffle(mentions)

    # --- actions -------------------------------------------------------------
    actions, action_lines = [], []
    action_templates = T.FORMAL_ACTION_TEMPLATES if formal else T.ACTION_TEMPLATES
    if style in ("narrative", "email"):
        action_templates = [t for t in action_templates if not t.startswith(("- ", "AI:"))]
    planned = []
    for code, sev in severities.items():
        n_act = 0 if rng.random() < (0.35 if formal else 0.15) else (rng.choice([1, 2]) if sev != "minor" else 1)
        for action in rng.sample(T.ISSUE_ACTIONS[code], n_act):
            if formal:
                owner = ("The PI" if code == "PI_OVERSIGHT" else rng.choice(["The pharmacist", "The SC"])
                         if code in ("IP_ACCOUNTABILITY", "TEMP_EXCURSION") else rng.choice(T.FORMAL_OWNERS))
            elif code == "PI_OVERSIGHT":
                owner = rng.choice([T.PI_OWNER, f"Dr. {pi_last}"])
            elif code in ("IP_ACCOUNTABILITY", "TEMP_EXCURSION"):
                owner = rng.choice(T.PHARMACY_OWNERS + T.SITE_OWNERS[:2])
            else:
                owner = rng.choice(T.SITE_OWNERS)
            planned.append((owner, action))
    if rng.random() < ((0.25 if formal else 0.5) if severities else (0.15 if formal else 0.3)):
        cra_owner = "The CRA" if formal else rng.choice(T.CRA_OWNERS + [f"CRA {cra_last}"])
        planned.append((cra_owner, rng.choice(T.GENERIC_ACTIONS)))
    for owner, action in planned:
        tmpl = S.pick(action_templates)
        due = None
        if "[[DUE:due]]" in tmpl:
            if rng.random() < 0.7:
                due = format_date(visit_date + timedelta(days=rng.randint(7, 45)), date_fmt)
            else:
                due = rng.choice(T.FORMAL_RELATIVE_DUES if formal else T.RELATIVE_DUES)
        vals = {"owner": owner, "action": action, "action_cap": _cap(action), "due": due}
        txt, sp = render(tmpl, vals)
        txt = _cap(txt)
        action_lines.append((txt, sp))
        got = {lab: txt[s:e] for lab, s, e in sp}
        actions.append({"action": got["ACTION"], "owner": got["OWNER"], "due": got.get("DUE")})

    risk_word = {"low": "Low", "medium": rng.choice(["Medium", "Moderate"]), "high": "High"}[risk]

    # --- assemble ------------------------------------------------------------
    nb = NoteBuilder()

    def add_t(tmpl: str, sep: str = "\n", extra: dict | None = None):
        txt, sp = render(tmpl, {**values, **(extra or {})})
        nb.add(txt, sp, sep)

    def distractor(sep: str):
        if rng.random() < 0.35:
            add_t(S.pick(T.DISTRACTORS), sep, _sentence_values(rng, ctx))

    if style == "field":
        add_t(S.pick(T.FIELD_HEADERS))
        if rng.random() < 0.75:
            add_t(S.pick(T.FIELD_PEOPLE))
        if vt != "SIV" and rng.random() < 0.85:
            add_t(S.pick(T.FIELD_ENROLLMENT))
        for _g, txt, sp, _c in mentions:
            bullet = "- " if rng.random() < 0.7 else ""
            nb.add(bullet + txt, [(lab, s + len(bullet), e + len(bullet)) for lab, s, e in sp])
        distractor("\n")
        for txt, sp in action_lines:
            nb.add(txt, sp)
    elif style == "report":
        add_t(S.pick(T.REPORT_OPENINGS))
        if vt != "SIV":
            add_t(S.pick(T.REPORT_ENROLLMENT), "\n\n")
        for group, headings in T.REPORT_SECTIONS.items():
            sents = [(_cap(txt), sp) for g, txt, sp, _c in mentions if g == group]
            if not sents and rng.random() < 0.6:
                continue
            nb.add(rng.choice(headings), [], "\n\n")
            if not sents:
                nb.add(rng.choice(T.NO_FINDINGS), [], "\n")
                continue
            first = True
            for txt, sp in sents:
                txt = txt if txt.endswith(".") else txt + "."
                nb.add(txt, sp, "\n" if first else " ")
                first = False
        distractor("\n\n")
        if rng.random() < 0.15:
            nb.add(rng.choice(T.RISK_LINES).format(risk_word=risk_word), [], "\n\n")
        if action_lines:
            nb.add(rng.choice(T.ACTION_HEADERS), [], "\n\n")
            for txt, sp in action_lines:
                nb.add(txt, sp, "\n")
    elif style == "narrative":
        add_t(S.pick([o for o in T.REPORT_OPENINGS if "\n" not in o]))
        if vt != "SIV":
            add_t(S.pick(T.REPORT_ENROLLMENT), " ")
        for _g, txt, sp, _c in mentions:
            txt = _cap(txt) if txt.endswith(".") else _cap(txt) + "."
            nb.add(txt, sp, " ")
        distractor(" ")
        sep = "\n\n"
        for txt, sp in action_lines:
            nb.add(txt if txt.endswith(".") else txt + ".", sp, sep)
            sep = " "
    elif style == "formal":
        _assemble_formal(rng, nb, values, vt, mentions, action_lines, ctx, S)
    else:  # email (held out)
        add_t(S.pick(T.EMAIL_OPENINGS))
        if vt != "SIV":
            add_t(S.pick(T.EMAIL_ENROLLMENT), " ")
        if mentions:
            nb.add(rng.choice(["A few things came up:", "Main points:", "Findings:"]), [], "\n\n")
            for _g, txt, sp, _c in mentions:
                nb.add("* " + txt, [(lab, s + 2, e + 2) for lab, s, e in sp], "\n")
        if action_lines:
            nb.add("Next steps:", [], "\n\n")
            for txt, sp in action_lines:
                nb.add(txt, sp, "\n")
        add_t(S.pick(T.EMAIL_SIGNOFFS), "\n\n")

    return _finalize(nb, {
        "style": style, "unseen": unseen, "risk": risk, "issues": sorted(severities, key=V1_ISSUE_CODES.index),
        "severities": severities, "inactive": inactive, "actions": actions,
        "meta": {"visit_type": vt, "visit_date": visit_date.isoformat(), "site_id": str(site_id),
                 "monitor": values["cra"], "pi": values["pi"], "screened": None if vt == "SIV" else ns,
                 "enrolled": None if vt == "SIV" else ne},
    })


def _assemble_formal(rng: random.Random, nb: NoteBuilder, values: dict, vt: str, mentions: list, action_lines: list,
                     ctx: dict, S: Sampler) -> None:
    """Long, formal CRA report: header block, then prose paragraphs or topic bullets."""
    title = S.pick(T.FORMAL_VISIT_TITLES[vt])
    label_part, _, suffix = title.partition(" (")
    suffix = f" ({suffix}" if suffix else ""
    site_val = rng.choice([f"Site {ctx['site_id']}", str(ctx["site_id"])])
    vals = {**values, "vt_f": label_part, "site_f": site_val}
    header = [
        "Monitor: [[MONITOR:cra]]" + rng.choice(T.FORMAL_CREDENTIALS),
        "Visit Type: [[VISIT_TYPE:vt_f]]" + suffix,
        rng.choice(["Date", "Visit Date", "Date of Visit"]) + ": [[VISIT_DATE:date]]",
    ]
    if rng.random() < 0.5:
        header.insert(rng.randint(0, 3), "Site: [[SITE:site_f]]")
    if rng.random() < 0.4:
        header.append("Principal Investigator: [[PI:pi]]")
    if rng.random() < 0.3:
        header[0], header[1] = header[1], header[0]
    for line in header:
        txt, sp = render(line, vals)
        nb.add(txt, sp)
    if rng.random() < 0.7:
        nb.add("Notes:", [])

    # body items: (topic, text, spans)
    items = [(T.FORMAL_TOPICS[code], txt, sp) for _g, txt, sp, code in mentions]
    seen_neutral, held_neutral = T.split_variants(T.FORMAL_NEUTRAL)
    pool = held_neutral if (S.unseen and held_neutral) else seen_neutral
    for tmpl in rng.sample(pool, min(len(pool), rng.randint(1, 3))):
        items.append(("General", render(tmpl, _sentence_values(rng, ctx))[0], []))
    if rng.random() < 0.25:
        items.append(("Action Item", S.pick(T.FORMAL_COMPLETED), []))
    if vt != "SIV" and rng.random() < 0.6:
        txt, sp = render(S.pick(T.FORMAL_ENROLLMENT), vals)
        if txt.startswith("Subject Status: "):
            cut = len("Subject Status: ")
            txt, sp = txt[cut:], [(lab, s - cut, e - cut) for lab, s, e in sp]
        items.append(("Subject Status", txt, sp))
    rng.shuffle(items)

    if rng.random() < 0.5:  # topic bullets
        for topic, txt, sp in items:
            prefix = f"* {topic}: "
            nb.add(prefix + txt, [(lab, s + len(prefix), e + len(prefix)) for lab, s, e in sp])
        for txt, sp in action_lines:
            nb.add("* " + txt, [(lab, s + 2, e + 2) for lab, s, e in sp])
    else:  # prose paragraphs
        n_par = min(len(items), rng.randint(1, 3))
        cuts = sorted(rng.sample(range(1, len(items)), n_par - 1)) if n_par > 1 else []
        start = 0
        for cut in cuts + [len(items)]:
            first = True
            for _topic, txt, sp in items[start:cut]:
                nb.add(txt, sp, "\n" if first else " ")
                first = False
            start = cut
        first = True
        for txt, sp in action_lines:
            nb.add(txt if txt.endswith(".") else txt + ".", sp, "\n" if first else " ")
            first = False


def _finalize(nb: NoteBuilder, row: dict) -> dict | None:
    text = nb.text
    tokens = tokenize(text)
    tags = char_spans_to_bio(tokens, nb.spans)
    # sanity check: every labelled span must line up with whole tokens
    for _label, cs, ce in nb.spans:
        covered = [t for t in tokens if t.start >= cs and t.end <= ce]
        if not covered or covered[0].start != cs or covered[-1].end != ce:
            return None
    return {"text": text, "tokens": [t.text for t in tokens], "tags": tags,
            "spans": [{"label": lab, "char_start": s, "char_end": e, "text": text[s:e]} for lab, s, e in nb.spans],
            **row}


def generate(n: int, seed: int, unseen: bool = False, prefix: str = "note") -> list[dict]:
    rng = random.Random(seed)
    rows, dropped = [], 0
    while len(rows) < n:
        row = sample_note(rng, unseen=unseen)
        if row is None:
            dropped += 1
            continue
        row = {"id": f"{prefix}-{len(rows):05d}", **row}
        rows.append(row)
    if dropped:
        print(f"[{prefix}] dropped {dropped} notes with misaligned spans")
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> str:
    risk = Counter(r["risk"] for r in rows)
    styles = Counter(r["style"] for r in rows)
    issues = Counter(c for r in rows for c in r["issues"])
    lengths = sorted(len(r["tokens"]) if "tokens" in r else len(r["text"].split()) for r in rows)
    return (f"n={len(rows)} risk={dict(risk)} styles={dict(styles)}\n"
            f"  words: median={lengths[len(lengths) // 2]} p95={lengths[int(len(lengths) * .95)]} max={lengths[-1]}\n"
            f"  issue counts: {dict(issues)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--n-unseen", type=int, default=1500)
    ap.add_argument("--n-corpus", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    out = Path(args.out)
    splits = {
        "train": generate(args.n_train, args.seed, prefix="train"),
        "val": generate(args.n_val, args.seed + 1, prefix="val"),
        "test": generate(args.n_test, args.seed + 2, prefix="test"),
        "test_unseen": generate(args.n_unseen, args.seed + 3, unseen=True, prefix="unseen"),
        "corpus": generate(args.n_corpus, args.seed + 4, prefix="visit"),
    }
    # the search corpus ships with the app, so keep only what search needs
    splits["corpus"] = [{k: r[k] for k in ("id", "text", "risk", "issues", "meta", "style")} for r in splits["corpus"]]
    for name, rows in splits.items():
        if rows:
            write_jsonl(rows, out / f"{name}.jsonl")
            print(f"{name:<12} {summarize(rows)}")


if __name__ == "__main__":
    main()
