"""
The search systems we compare. Each one has a `name` and a `search(query, k) -> [job_id, ...]` method,
so labeling and evaluation treat them all the same way. Step 6 adds its system to SYSTEMS here.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app import db, vector_index
from app.embedding import encode

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or our that the this to was we were will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that isn't a letter or digit, drop very common words."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Search:
    """Keyword baseline: ranks jobs by how often they contain the query's words, weighted by how rare each word is."""
    name = "bm25"

    def __init__(self, conn):
        rows = conn.execute("SELECT job_id, title, description FROM jobs ORDER BY job_id").fetchall()
        self.ids = np.array([r["job_id"] for r in rows])
        # BM25 has no length limit, so it gets the FULL description (its natural best setup).
        self.bm25 = BM25Okapi([tokenize(r["title"] + " " + r["description"]) for r in rows])

    def search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [int(self.ids[i]) for i in top if scores[i] > 0]  # no shared word -> no result, as in real keyword search


class EmbeddingSearch:
    """Our semantic search: the same index and model the API uses."""
    name = "embedding"

    def __init__(self, index_path: str = "data/jobs.index"):
        self.index = vector_index.load(index_path)

    def search(self, query: str, k: int) -> list[int]:
        return [job_id for job_id, _ in vector_index.search(self.index, encode([query]), k)]


def load_systems() -> list:
    conn = db.connect()
    return [BM25Search(conn), EmbeddingSearch()]
