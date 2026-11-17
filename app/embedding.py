"""
Everything about turning text into vectors lives here, so the build script,
the API and the evaluation all embed text in exactly the same way.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The model only reads ~256 tokens (~1,000 characters of English).
# Anything longer is silently cut off, so we cut it ourselves, on purpose.
DESC_CHARS = 1000


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes seconds; encoding takes milliseconds)."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


def job_to_text(title: str, description: str) -> str:
    """The exact text we embed for one job: the title first (most informative), then the start of the description."""
    return f"{title}. {description[:DESC_CHARS]}"


def encode(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Turn a list of texts into a (len(texts), 384) float32 array of unit-length vectors."""
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # unit length -> inner product == cosine similarity
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype("float32")  # FAISS requires float32
