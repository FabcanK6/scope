"""Load the hand-written (``handwritten.txt``) and realistic (``realistic.txt``) evaluation notes as dataset rows."""

from __future__ import annotations

import re
from pathlib import Path

from scope.text import Span, bio_to_spans, char_spans_to_bio, tokenize

HANDWRITTEN_PATH = Path(__file__).with_name("handwritten.txt")
REALISTIC_PATH = Path(__file__).with_name("realistic.txt")
_MARK_RE = re.compile(r"\[\[(.+?)\|([A-Z_]+)\]\]")


def parse_marked(marked: str) -> tuple[str, list[tuple[str, int, int]]]:
    """'[[Site 104|SITE]] visit' -> ('Site 104 visit', [('SITE', 0, 8)])"""
    out, spans, pos, length = [], [], 0, 0
    for m in _MARK_RE.finditer(marked):
        lit = marked[pos:m.start()]
        out.append(lit)
        length += len(lit)
        value, label = m.group(1), m.group(2)
        spans.append((label, length, length + len(value)))
        out.append(value)
        length += len(value)
        pos = m.end()
    out.append(marked[pos:])
    return "".join(out), spans


def load_realistic() -> list[dict]:
    return load_handwritten(REALISTIC_PATH, style="realistic")


def load_handwritten(path: str | Path = HANDWRITTEN_PATH, style: str = "handwritten") -> list[dict]:
    from scope.record import assemble_actions

    raw = Path(path).read_text()
    raw = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
    blocks = re.split(r"^=== ", raw, flags=re.M)[1:]
    rows = []
    for block in blocks:
        header, _, body = block.partition("\n---\n")
        lines = header.strip().splitlines()
        note_id = lines[0].strip()
        fields = dict(line.split(":", 1) for line in lines[1:] if ":" in line)
        issues = [c.strip() for c in fields.get("issues", "").split(",") if c.strip()]
        text, char_spans = parse_marked(body.strip("\n"))
        tokens = tokenize(text)
        tags = char_spans_to_bio(tokens, char_spans)
        spans: list[Span] = bio_to_spans(tokens, tags, text)
        rows.append({
            "id": note_id, "text": text, "tokens": [t.text for t in tokens], "tags": tags,
            "risk": fields["risk"].strip(), "issues": issues, "style": style, "unseen": True,
            "actions": [{k: a[k] for k in ("action", "owner", "due")} for a in assemble_actions(text, spans)],
            "spans": [{"label": lab, "char_start": s, "char_end": e, "text": text[s:e]} for lab, s, e in char_spans],
        })
    return rows
