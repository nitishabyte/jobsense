"""
Step 5b: score every system against your labels.

Metrics
  precision@5 : of the 5 jobs shown, what fraction are relevant? (fewer than 5 results -> missing slots count as misses)
  MRR@5       : 1 / rank of the first relevant job (1.0 = top result relevant, 0 = none in the top 5)
  latency     : median time to answer one query (ranking only, no database lookup), over several repeats

Usage (from the repo root):
    python -m evaluation.evaluate
"""
import json
import statistics
import time
from pathlib import Path

from evaluation.systems import load_systems

QUERIES = Path("evaluation/queries.json")
LABELS = Path("evaluation/labels.json")
RESULTS = Path("evaluation/results.md")
K = 5
LATENCY_REPEATS = 5


def precision_at_k(rels: list[int], k: int = K) -> float:
    return sum(rels[:k]) / k


def mrr_at_k(rels: list[int], k: int = K) -> float:
    for rank, r in enumerate(rels[:k], start=1):
        if r:
            return 1 / rank
    return 0.0


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    queries = json.loads(QUERIES.read_text())
    labels = json.loads(LABELS.read_text())
    systems = load_systems()

    per_query = {s.name: {} for s in systems}   # system -> query id -> (p@5, mrr)
    latency = {s.name: [] for s in systems}
    unlabeled = []

    for s in systems:
        s.search("warm up", K)  # first call loads/caches things; don't time it
        for q in queries:
            top = s.search(q["query"], K)
            q_labels = labels.get(q["id"], {})
            missing = [j for j in top if str(j) not in q_labels]
            if missing:
                unlabeled.append((s.name, q["id"], missing))
                continue
            rels = [q_labels[str(j)] for j in top]
            per_query[s.name][q["id"]] = (precision_at_k(rels), mrr_at_k(rels))
            for _ in range(LATENCY_REPEATS):
                start = time.perf_counter()
                s.search(q["query"], K)
                latency[s.name].append((time.perf_counter() - start) * 1000)

    if unlabeled:
        print("Some results have no label yet - run `python -m evaluation.label` first:")
        for name, qid, ids in unlabeled:
            print(f"  {name} / {qid}: {len(ids)} unlabeled")
        raise SystemExit(1)

    # ---------- Summary table ----------
    by_type = lambda t: [q["id"] for q in queries if q["type"] == t]
    lines = [
        f"Evaluated on {len(queries)} queries ({len(by_type('title'))} title-style, {len(by_type('natural'))} natural-language), "
        f"labels: {sum(len(v) for v in labels.values())} judged (query, job) pairs.\n",
        "| System | P@5 (all) | P@5 title-style | P@5 natural-language | MRR@5 | Median latency |",
        "|---|---|---|---|---|---|",
    ]
    for s in systems:
        pq = per_query[s.name]
        lines.append(
            f"| {s.name} | {mean(p for p, _ in pq.values()):.3f} "
            f"| {mean(pq[i][0] for i in by_type('title')):.3f} "
            f"| {mean(pq[i][0] for i in by_type('natural')):.3f} "
            f"| {mean(m for _, m in pq.values()):.3f} "
            f"| {statistics.median(latency[s.name]):.1f} ms |"
        )

    # ---------- Per-query table ----------
    names = [s.name for s in systems]
    lines += ["", "Per-query precision@5:", "", "| Query | Type | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for q in queries:
        lines.append(f"| {q['query']} | {q['type']} | " + " | ".join(f"{per_query[n][q['id']][0]:.1f}" for n in names) + " |")

    # ---------- Head-to-head: each system vs the first one (the baseline) ----------
    base = names[0]
    lines.append("")
    for n in names[1:]:
        wins = sum(per_query[n][q["id"]][0] > per_query[base][q["id"]][0] for q in queries)
        losses = sum(per_query[n][q["id"]][0] < per_query[base][q["id"]][0] for q in queries)
        lines.append(f"{n} vs {base}: better on {wins} queries, worse on {losses}, tied on {len(queries) - wins - losses}.")

    report = "\n".join(lines)
    print(report)
    RESULTS.write_text(report + "\n")
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
