"""
Quick manual check of the index from the terminal.

Usage:
    python -m scripts.try_search "backend role working with AI" --k 5
"""
import argparse
import time

import pandas as pd

from app.embedding import encode
from app import vector_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--csv", default="data/jobs_clean.csv")
    p.add_argument("--index", default="data/jobs.index")
    args = p.parse_args()

    jobs = pd.read_csv(args.csv).set_index("job_id")  # look up a job by its id
    index = vector_index.load(args.index)
    encode(["warm-up"])  # first call loads the model; don't count that in the query time

    start = time.perf_counter()
    results = vector_index.search(index, encode([args.query]), args.k)
    ms = (time.perf_counter() - start) * 1000

    print(f'query: "{args.query}"  ({ms:.0f} ms)\n')
    for rank, (job_id, score) in enumerate(results, start=1):
        job = jobs.loc[job_id]
        print(f"{rank}. [{score:.3f}] {job['title']} | {job['company_name']} | {job['location']}")
        print(f"   {job['description'][:150]}...\n")


if __name__ == "__main__":
    main()
