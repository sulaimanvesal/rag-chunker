"""Demo: compare chunking strategies on a sample document. No API keys needed."""
from rag_chunker import compare_strategies, chunk_recursive

SAMPLE = open(__file__.replace("demo.py", "sample.txt"), encoding="utf-8").read() if False else None

TEXT = (
    "Embeddings turn text into vectors. Similar meanings land close together in that space. "
    "A retriever searches those vectors for the passages closest to a question.\n\n"
    "Chunking happens before embedding. If a chunk mixes two topics, its vector points between them "
    "and retrieves poorly for both. If a chunk is too small, it loses the sentence that gave it meaning.\n\n"
    "Start with recursive chunking at 400-600 characters for prose. Measure retrieval on your own "
    "questions before tuning further. Most gains come from clean source text, not exotic strategies."
)

if __name__ == "__main__":
    print("=== Strategy comparison (max_chars=200) ===")
    for name, s in compare_strategies(TEXT, max_chars=200).items():
        print(f"{name:12} chunks={s['count']} avg_chars={s['avg_chars']} tokens~{s['total_tokens_est']}")
    print("\n=== Recursive chunks ===")
    for c in chunk_recursive(TEXT, max_chars=200):
        print(f"\n--- chunk {c.index} ({len(c.text)} chars, ~{c.token_estimate} tokens) ---")
        print(c.text)
    print("\nDemo OK")
