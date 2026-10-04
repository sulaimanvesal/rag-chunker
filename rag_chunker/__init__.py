"""rag-chunker: chunking toolkit for RAG pipelines."""
from .core import (
    Chunk,
    chunk_fixed,
    chunk_paragraphs,
    chunk_recursive,
    chunk_sentences,
    compare_strategies,
    estimate_tokens,
)

__all__ = [
    "Chunk",
    "chunk_fixed",
    "chunk_paragraphs",
    "chunk_recursive",
    "chunk_sentences",
    "compare_strategies",
    "estimate_tokens",
]
__version__ = "0.1.0"
