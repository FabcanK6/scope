"""Command line: analyze one note.

    python -m scope.cli "IMV Site 104 12-Mar-2026. 23 queries open > 60 days. CRC to close queries by 26-Mar-2026."
    python -m scope.cli --file note.txt --json
    python -m scope.cli --file note.txt --backend rules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("note", nargs="?", help="note text (or use --file, or pipe text in)")
    ap.add_argument("--file", help="read the note from this file")
    ap.add_argument("--model", default="models/scope-bert")
    ap.add_argument("--backend", choices=["hybrid", "bert", "rules"], default="hybrid")
    ap.add_argument("--json", action="store_true", help="print the full visit record as JSON")
    args = ap.parse_args(argv)

    if args.file:
        text = Path(args.file).read_text()
    elif args.note:
        text = args.note
    else:
        text = sys.stdin.read()
    if not text.strip():
        ap.error("no note text given")

    from scope.predict import load_parser
    from scope.record import audit_summary

    parser = load_parser(args.model, backend=args.backend)
    rec = parser.analyze(text)
    print(json.dumps(rec, indent=2) if args.json else audit_summary(rec))


if __name__ == "__main__":
    main()
