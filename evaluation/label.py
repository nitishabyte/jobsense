"""
Step 5a: label which jobs are relevant to each test query.

For every query we take the top-5 from EVERY system, merge them into one "pool", shuffle it,
and show the jobs one by one WITHOUT saying which system found them or their score (blind labeling).
Answers are saved after every keypress, so you can quit and resume any time.
Re-running later (e.g. after adding a system in Step 6) only asks about jobs not yet labeled.

Usage (from the repo root):
    python -m evaluation.label
"""
import json
import random
from pathlib import Path

from app import db
from evaluation.systems import load_systems

QUERIES = Path("evaluation/queries.json")
LABELS = Path("evaluation/labels.json")
POOL_DEPTH = 5  # we report precision@5, so the top-5 of each system is all we need labeled


def save(labels: dict) -> None:
    LABELS.write_text(json.dumps(labels, indent=1, sort_keys=True))


def main() -> None:
    queries = json.loads(QUERIES.read_text())
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    systems = load_systems()
    conn = db.connect()

    # Build the to-do list: pooled jobs per query that don't have a label yet.
    todo = []
    for q in queries:
        pool = {job_id for s in systems for job_id in s.search(q["query"], POOL_DEPTH)}
        unlabeled = sorted(j for j in pool if str(j) not in labels.get(q["id"], {}))
        random.Random(q["id"]).shuffle(unlabeled)  # hide each system's ranking order (reproducible shuffle)
        todo += [(q, j) for j in unlabeled]

    print(f"{len(todo)} jobs to label across {len(queries)} queries. Keys: y = relevant, n = not, o = show full text, q = save & quit\n")
    for n, (q, job_id) in enumerate(todo, start=1):
        job = db.fetch_jobs(conn, [job_id])[job_id]
        print("=" * 90)
        print(f'[{n}/{len(todo)}]  QUERY {q["id"]}: "{q["query"]}"')
        print(f"RELEVANT IF: {q['relevant_if']}")
        print("-" * 90)
        print(f"TITLE:    {job['title']}")
        print(f"COMPANY:  {job['company']}  |  {job['location']}  |  {job['categories']}")
        print(f"\n{job['description'][:700]}...")
        while True:
            answer = input("\nRelevant? [y/n/o/q]: ").strip().lower()
            if answer == "o":
                print("\n" + job["description"][:4000])
            elif answer in ("y", "n"):
                labels.setdefault(q["id"], {})[str(job_id)] = 1 if answer == "y" else 0
                save(labels)
                break
            elif answer == "q":
                print(f"Saved. {n - 1} labeled this session; run again to continue.")
                return
    print("\nAll done. Labels saved to", LABELS)


if __name__ == "__main__":
    main()
