"""Word tokenization with character offsets, and BIO helpers.

Training data and inference both go through :func:`tokenize`, so the word
boundaries the model sees at inference match the ones it was trained on.
Offsets let the app highlight spans in the original note, line breaks included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'/.:][A-Za-z0-9]+)*%?|[^\sA-Za-z0-9]")


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # character offset, inclusive
    end: int    # character offset, exclusive


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def words(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]


@dataclass
class Span:
    label: str
    start: int        # word index, inclusive
    end: int          # word index, exclusive
    text: str
    char_start: int = -1
    char_end: int = -1

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.label, self.start, self.end)

    def to_dict(self) -> dict:
        return {"label": self.label, "text": self.text, "start": self.start, "end": self.end,
                "char_start": self.char_start, "char_end": self.char_end}


def char_spans_to_bio(tokens: list[Token], char_spans: list[tuple[str, int, int]]) -> list[str]:
    """Tag every token that lies fully inside a character span."""
    tags = ["O"] * len(tokens)
    for label, cs, ce in sorted(char_spans, key=lambda s: s[1]):
        first = True
        for i, tok in enumerate(tokens):
            if tok.start >= cs and tok.end <= ce:
                tags[i] = f"{'B' if first else 'I'}-{label}"
                first = False
    return tags


def bio_to_spans(tokens: list[Token] | list[str], tags: list[str], text: str | None = None) -> list[Span]:
    """Decode BIO tags into spans. A stray ``I-X`` is treated as the start of a span.

    With :class:`Token` objects (and the source ``text``) the span text is the
    exact substring of the note, and character offsets are filled in.
    """
    spans: list[Span] = []
    cur_label, cur_start = None, None

    def close(end: int) -> None:
        toks = tokens[cur_start:end]
        if toks and isinstance(toks[0], Token):
            cs, ce = toks[0].start, toks[-1].end
            span_text = text[cs:ce] if text is not None else " ".join(t.text for t in toks)
            spans.append(Span(cur_label, cur_start, end, span_text, cs, ce))
        else:
            spans.append(Span(cur_label, cur_start, end, " ".join(toks)))

    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and label == cur_label
        if cur_label is not None and not continues:
            close(i)
            cur_label, cur_start = None, None
        if prefix in ("B", "I") and not continues:
            cur_label, cur_start = label, i
    return spans


def spans_to_bio(n: int, spans: list[tuple[str, int, int]]) -> list[str]:
    tags = ["O"] * n
    for label, s, e in spans:
        tags[s] = f"B-{label}"
        for j in range(s + 1, e):
            tags[j] = f"I-{label}"
    return tags


def sentences(text: str) -> list[tuple[int, int]]:
    """Rough sentence/line boundaries as character ranges (used by the rule baseline)."""
    out = []
    for line in re.finditer(r"[^\n]+", text):
        base, chunk = line.start(), line.group(0)
        start = 0
        for m in _SENT_BREAK.finditer(chunk):
            if chunk[start:m.start()].strip():
                out.append((base + start, base + m.start()))
            start = m.end()
        if chunk[start:].strip():
            out.append((base + start, base + len(chunk)))
    return out


# a sentence ends at . ; ! ? followed by whitespace, unless the period belongs to a title like "Dr."
_SENT_BREAK = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bvs\.)(?<=[.;!?])\s+")
