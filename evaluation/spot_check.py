"""
Check the LLM judge: you label a random sample of already-labeled pairs WITHOUT seeing the existing label,
then we measure how often you and the judge agree.

Usage (from the repo root):
    python -m evaluation.spot_check            # label the sample (resumable: q to quit)
    python -m evaluation.spot_check --report   # show agreement
"""
import argparse
import json
import random
from pathlib import Path

from app import db

QUERIES = Path("evaluation/queries.json")
LABELS = Path("evaluation/labels.json")
CHECK = Path("evaluation/spot_check.json")
SAMPLE_SIZE = 30
SEED = 7  # fixed, so the sample is chosen before anyone sees which pairs it contains


def sample_pairs(labels: dict) -> list[tuple[str, str]]:
    pairs = sorted((qid, jid) for qid, jobs in labels.items() for jid in jobs)
    return random.Random(SEED).sample(pairs, min(SAMPLE_SIZE, len(pairs)))


def report(labels: dict, mine: dict) -> None:
    pairs = [p for p in sample_pairs(labels) if f"{p[0]}:{p[1]}" in mine]
    if not pairs:
        raise SystemExit("Nothing spot-checked yet.")
    judge = [labels[q][j] for q, j in pairs]
    human = [mine[f"{q}:{j}"] for q, j in pairs]
    n = len(pairs)
    agree = sum(a == b for a, b in zip(judge, human))
    # Cohen's kappa: agreement corrected for what two random labelers with the same yes-rates would reach by chance.
    p_yes_j, p_yes_h = sum(judge) / n, sum(human) / n
    p_chance = p_yes_j * p_yes_h + (1 - p_yes_j) * (1 - p_yes_h)
    kappa = (agree / n - p_chance) / (1 - p_chance) if p_chance < 1 else 1.0
    print(f"checked {n} pairs: agreement {agree}/{n} = {agree / n:.0%}, Cohen's kappa = {kappa:.2f}")
    for (q, j), a, b in zip(pairs, judge, human):
        if a != b:
            print(f"  disagree: {q} job {j}  judge={a}  you={b}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    labels = json.loads(LABELS.read_text())
    mine = json.loads(CHECK.read_text()) if CHECK.exists() else {}
    if args.report:
        return report(labels, mine)

    queries = {q["id"]: q for q in json.loads(QUERIES.read_text())}
    conn = db.connect()
    todo = [(q, j) for q, j in sample_pairs(labels) if f"{q}:{j}" not in mine]
    print(f"{len(todo)} to check. Keys: y = relevant, n = not, o = full text, q = save & quit\n")
    for n, (qid, jid) in enumerate(todo, start=1):
        q, job = queries[qid], db.fetch_jobs(conn, [int(jid)])[int(jid)]
        print("=" * 90)
        print(f'[{n}/{len(todo)}]  QUERY {qid}: "{q["query"]}"\nRELEVANT IF: {q["relevant_if"]}')
        print("-" * 90)
        print(f"TITLE:    {job['title']}\nCOMPANY:  {job['company']}  |  {job['location']}  |  {job['categories']}")
        print(f"\n{job['description'][:700]}...")
        while True:
            a = input("\nRelevant? [y/n/o/q]: ").strip().lower()
            if a == "o":
                print("\n" + job["description"][:4000])
            elif a in ("y", "n"):
                mine[f"{qid}:{jid}"] = 1 if a == "y" else 0
                CHECK.write_text(json.dumps(mine, indent=1, sort_keys=True))
                break
            elif a == "q":
                return
    print("\nDone. Now run: python -m evaluation.spot_check --report")


if __name__ == "__main__":
    main()
