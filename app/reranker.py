"""
Step 6: cross-encoder re-ranking.

The embedding model (a bi-encoder) turns the query and each job into vectors SEPARATELY, then compares them.
A cross-encoder reads the query and one job TOGETHER in a single pass, so every query word can attend to every
job word. That is much more accurate, but far too slow to run on all 5,000 jobs, so we use it only to re-order
the embedding search's top candidates ("retrieve, then re-rank").
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"  # small (6 layers), trained to rank passages for search queries
RERANK_DESC_CHARS = 2000  # the cross-encoder reads up to 512 tokens (~2,000 chars), about twice what the embedder sees


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")


def rerank_text(title: str, description: str) -> str:
    return f"{title}. {description[:RERANK_DESC_CHARS]}"


def rerank(query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
    """candidates: [(job_id, text), ...] -> [(job_id, relevance_score), ...] best first."""
    if not candidates:
        return []
    scores = get_reranker().predict([(query, text) for _, text in candidates])
    ranked = sorted(zip((job_id for job_id, _ in candidates), scores), key=lambda pair: -pair[1])
    return [(job_id, float(score)) for job_id, score in ranked]
