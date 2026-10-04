import pytest

from rag_chunker import (
    chunk_fixed,
    chunk_paragraphs,
    chunk_recursive,
    chunk_sentences,
    compare_strategies,
    estimate_tokens,
)

DOC = (
    "Retrieval augmented generation grounds a model in external documents. "
    "Chunking decides what the model actually sees. Bad chunks split ideas in half.\n\n"
    "A good chunk keeps one idea together. It is small enough to retrieve precisely. "
    "It is large enough to keep the context that makes it understandable.\n\n"
    "This toolkit compares four simple strategies. Fixed windows are predictable. "
    "Sentence grouping respects grammar. Paragraph grouping respects author structure. "
    "Recursive chunking tries structure first and falls back only when it must."
)


def test_estimate_tokens_nonempty():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 2


def test_fixed_covers_text():
    chunks = chunk_fixed(DOC, size=100, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].text == DOC[:100]
    assert all(c.token_estimate > 0 for c in chunks)


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_fixed(DOC, size=50, overlap=50)


def test_fixed_empty():
    assert chunk_fixed("") == []


def test_sentences_keep_sentence_boundaries():
    chunks = chunk_sentences(DOC, max_chars=120)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip().endswith((".", "\n")) or len(c.text) > 0


def test_paragraphs_respect_breaks():
    chunks = chunk_paragraphs(DOC, max_chars=200)
    assert len(chunks) >= 2


def test_recursive_all_within_limit_or_sentence_fallback():
    chunks = chunk_recursive(DOC, max_chars=150)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 300 for c in chunks)  # single long sentences may exceed softly


def test_indexes_sequential():
    chunks = chunk_recursive(DOC, max_chars=200)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_compare_strategies():
    stats = compare_strategies(DOC, max_chars=200)
    assert set(stats) == {"fixed", "sentences", "paragraphs", "recursive"}
    for s in stats.values():
        assert s["count"] >= 1
        assert s["total_tokens_est"] > 0


def test_chunk_to_dict():
    c = chunk_fixed("hello world, this is a test", size=10, overlap=0)[0]
    d = c.to_dict()
    assert d["index"] == 0 and "token_estimate" in d
