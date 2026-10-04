"""CLI: python -m rag_chunker.cli <file> [--strategy recursive] [--max-chars 500] [--compare]"""
from __future__ import annotations

import argparse
import json
import sys

from .core import STRATEGIES, compare_strategies


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag-chunker", description="Chunk text for RAG pipelines")
    p.add_argument("file", help="Text file to chunk, or - for stdin")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="recursive")
    p.add_argument("--max-chars", type=int, default=500)
    p.add_argument("--compare", action="store_true", help="Compare all strategies")
    p.add_argument("--json", action="store_true", help="Output chunks as JSON")
    args = p.parse_args(argv)

    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()

    if args.compare:
        stats = compare_strategies(text, max_chars=args.max_chars)
        for name, s in stats.items():
            print(f"{name:12} chunks={s['count']:3}  avg_chars={s['avg_chars']:7}  max_chars={s['max_chars']:5}  tokens~{s['total_tokens_est']}")
        return 0

    if args.strategy == "fixed":
        chunks = STRATEGIES["fixed"](text, size=args.max_chars, overlap=50)
    elif args.strategy in ("sentences", "paragraphs"):
        chunks = STRATEGIES[args.strategy](text, max_chars=args.max_chars)
    else:
        chunks = STRATEGIES["recursive"](text, max_chars=args.max_chars)

    if args.json:
        print(json.dumps([c.to_dict() for c in chunks], indent=2))
    else:
        for c in chunks:
            preview = c.text.replace("\n", " ")[:100]
            print(f"[{c.index}] chars={len(c.text)} tokens~{c.token_estimate} :: {preview}")
        print(f"\nTotal: {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
