"""
Export the jobs that still need a relevance label, so they can be labeled outside this machine (e.g. by an LLM judge).
Same pooling as label.py: top-5 from every system, merged and shuffled, with NO system names or scores (blind).

Usage (from the repo root):
    python -m evaluation.export_pool
"""
import json
import random
from pathlib import Path

from app import db
from evaluation.systems import load_systems

QUERIES = Path("evaluation/queries.json")
LABELS = Path("evaluation/labels.json")
OUT = Path("evaluation/pool_to_label.jsonl")
POOL_DEPTH = 5
DESC_CHARS = 2500  # enough of the description to judge relevance


def main() -> None:
    queries = json.loads(QUERIES.read_text())
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    systems = load_systems()
    conn = db.connect()

    count = 0
    with OUT.open("w") as f:
        for q in queries:
            pool = {job_id for s in systems for job_id in s.search(q["query"], POOL_DEPTH)}
            unlabeled = sorted(j for j in pool if str(j) not in labels.get(q["id"], {}))
            random.Random(q["id"]).shuffle(unlabeled)
            jobs = db.fetch_jobs(conn, unlabeled)
            for job_id in unlabeled:
                j = jobs[job_id]
                f.write(json.dumps({
                    "query_id": q["id"], "query": q["query"], "relevant_if": q["relevant_if"],
                    "job_id": job_id, "title": j["title"], "company": j["company"],
                    "categories": j["categories"], "description": j["description"][:DESC_CHARS],
                }) + "\n")
                count += 1
    print(f"wrote {count} (query, job) pairs to {OUT}")


if __name__ == "__main__":
    main()
