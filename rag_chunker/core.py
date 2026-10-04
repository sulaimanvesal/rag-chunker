"""Core chunking strategies for RAG pipelines. Zero dependencies."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    text: str
    index: int
    start: int
    end: int
    token_estimate: int

    def to_dict(self):
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate: ~4 chars/token, at least word count//2.

    Good enough for budgeting and comparison without a tokenizer dependency.
    """
    if not text:
        return 0
    char_est = (len(text) + 3) // 4
    word_est = len(text.split())
    # Blend: real tokenizers sit between these for English prose.
    return max(1, (char_est + word_est) // 2 + word_est // 2) if word_est else char_est


def _make_chunks(pieces: list[tuple[str, int, int]]) -> list[Chunk]:
    return [
        Chunk(text=t, index=i, start=s, end=e, token_estimate=estimate_tokens(t))
        for i, (t, s, e) in enumerate(pieces)
    ]


def chunk_fixed(text: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split into fixed character windows with overlap."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    if not text:
        return []
    pieces = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if piece.strip():
            pieces.append((piece, start, start + len(piece)))
        if start + size >= len(text):
            break
    return _make_chunks(pieces)


def _split_sentences(text: str) -> list[tuple[str, int]]:
    parts = []
    for m in re.finditer(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$", text):
        s = m.group(0)
        if s.strip():
            parts.append((s, m.start()))
    return parts


def chunk_sentences(text: str, max_chars: int = 500, overlap_sentences: int = 1) -> list[Chunk]:
    """Group whole sentences until max_chars, optionally overlapping sentences."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    sents = _split_sentences(text)
    if not sents:
        return []
    pieces = []
    i = 0
    while i < len(sents):
        buf: list[str] = []
        start = sents[i][1]
        total = 0
        j = i
        while j < len(sents) and (total + len(sents[j][0]) <= max_chars or not buf):
            buf.append(sents[j][0])
            total += len(sents[j][0])
            j += 1
            # A single oversized sentence still forms its own chunk.
            if len(buf) == 1 and total > max_chars:
                break
        joined = "".join(buf)
        pieces.append((joined, start, start + len(joined)))
        if j >= len(sents):
            break
        i = max(j - overlap_sentences, i + 1)
    return _make_chunks(pieces)


def chunk_paragraphs(text: str, max_chars: int = 800) -> list[Chunk]:
    """Group paragraphs (blank-line separated) until max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    paras: list[tuple[str, int]] = []
    for m in re.finditer(r"\S.*?(?:\n\s*\n|\Z)", text, flags=re.S):
        s = m.group(0)
        if s.strip():
            paras.append((s, m.start()))
    if not paras:
        # Fall back to fixed chunking for text without paragraph breaks.
        return chunk_fixed(text, size=max_chars, overlap=0)
    pieces = []
    buf: list[str] = []
    start = paras[0][1]
    total = 0
    for para, pos in paras:
        if buf and total + len(para) > max_chars:
            joined = "".join(buf)
            pieces.append((joined, start, start + len(joined)))
            buf, total, start = [], 0, pos
        if not buf:
            start = pos
        buf.append(para)
        total += len(para)
    if buf:
        joined = "".join(buf)
        pieces.append((joined, start, start + len(joined)))
    return _make_chunks(pieces)


def chunk_recursive(text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Recursive strategy: paragraphs -> sentences -> fixed windows.

    Tries to keep natural boundaries, falling back to smaller units
    only when a unit exceeds max_chars.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    para_chunks = chunk_paragraphs(text, max_chars=max_chars)
    out: list[tuple[str, int, int]] = []
    for ch in para_chunks:
        if len(ch.text) <= max_chars:
            out.append((ch.text, ch.start, ch.end))
        else:
            for sub in chunk_sentences(ch.text, max_chars=max_chars):
                if len(sub.text) <= max_chars:
                    out.append((sub.text, ch.start + sub.start, ch.start + sub.end))
                else:
                    for fx in chunk_fixed(sub.text, size=max_chars, overlap=min(overlap, max_chars // 4)):
                        out.append((fx.text, ch.start + sub.start + fx.start, ch.start + sub.start + fx.end))
    return _make_chunks(out)


STRATEGIES = {
    "fixed": chunk_fixed,
    "sentences": chunk_sentences,
    "paragraphs": chunk_paragraphs,
    "recursive": chunk_recursive,
}


def compare_strategies(text: str, max_chars: int = 500) -> dict:
    """Run all strategies and return summary stats per strategy."""
    result = {}
    for name in STRATEGIES:
        if name == "fixed":
            chunks = chunk_fixed(text, size=max_chars, overlap=50)
        elif name == "sentences":
            chunks = chunk_sentences(text, max_chars=max_chars)
        elif name == "paragraphs":
            chunks = chunk_paragraphs(text, max_chars=max_chars)
        else:
            chunks = chunk_recursive(text, max_chars=max_chars)
        sizes = [len(c.text) for c in chunks]
        result[name] = {
            "count": len(chunks),
            "avg_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "max_chars": max(sizes) if sizes else 0,
            "total_tokens_est": sum(c.token_estimate for c in chunks),
        }
    return result
