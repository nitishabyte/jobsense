"""
Thin wrapper around FAISS so the rest of the code never touches FAISS directly.
"""
from pathlib import Path

import faiss
import numpy as np


def new_index(dim: int) -> faiss.Index:
    """
    IndexFlatIP   = exact search by inner product (= cosine, because our vectors are normalised).
    IndexIDMap2   = lets us store OUR job_id with each vector instead of FAISS's 0,1,2...
                    and supports adding/removing single vectors later without a rebuild.
    """
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))


def add(index: faiss.Index, vectors: np.ndarray, ids) -> None:
    index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))


def remove(index: faiss.Index, ids) -> int:
    """Remove vectors by job_id; returns how many were removed."""
    return index.remove_ids(np.asarray(ids, dtype="int64"))


def ids_in(index: faiss.Index) -> set[int]:
    """All job_ids currently stored in the index."""
    return set(faiss.vector_to_array(index.id_map).tolist())


def search(index: faiss.Index, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(job_id, score), ...] best first. FAISS pads with id -1 if it has fewer than k vectors."""
    scores, ids = index.search(query_vector, k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def save(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
