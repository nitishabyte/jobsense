"""
Quick manual check of the index from the terminal.
FAISS finds the best job_ids; SQLite supplies the details.

Usage:
    python -m scripts.try_search "backend role working with AI" --k 5
"""
import argparse
import time

from app import db, vector_index
from app.embedding import encode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index", default="data/jobs.index")
    args = p.parse_args()

    conn = db.connect()
    index = vector_index.load(args.index)
    encode(["warm-up"])  # first call loads the model; don't count that in the query time

    start = time.perf_counter()
    results = vector_index.search(index, encode([args.query]), args.k)
    jobs = db.fetch_jobs(conn, [job_id for job_id, _ in results])
    ms = (time.perf_counter() - start) * 1000

    print(f'query: "{args.query}"  ({ms:.0f} ms, embedding + FAISS + SQLite)\n')
    for rank, (job_id, score) in enumerate(results, start=1):
        job = jobs[job_id]
        print(f"{rank}. [{score:.3f}] {job['title']} | {job['company']} | {job['location']}")
        print(f"   categories: {job['categories']}")
        print(f"   {job['description'][:150]}...\n")


if __name__ == "__main__":
    main()
