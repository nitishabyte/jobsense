"""
Step 2: embed every cleaned job and build the FAISS index.

Usage (from the repo root):
    python -m scripts.build_index
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.embedding import DESC_CHARS, MODEL_NAME, encode, get_model, job_to_text
from app import vector_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/jobs_clean.csv")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()
    out = Path(args.out_dir)

    df = pd.read_csv(args.csv)
    print(f"jobs loaded: {len(df):,}")

    # --- How much text would the model throw away? (measure, don't guess) ---
    model = get_model()
    full_texts = (df["title"] + ". " + df["description"]).tolist()
    token_counts = [len(ids) for ids in model.tokenizer(full_texts, verbose=False)["input_ids"]]
    limit = model.max_seq_length
    over = sum(c > limit for c in token_counts)
    print(f"model token limit: {limit}")
    print(f"median tokens per full job text: {int(np.median(token_counts))}")
    print(f"jobs longer than the limit: {over:,} / {len(df):,} ({over / len(df):.1%})")

    # --- Embed ---
    texts = [job_to_text(t, d) for t, d in zip(df["title"], df["description"])]
    start = time.perf_counter()
    vectors = encode(texts, show_progress=True)
    secs = time.perf_counter() - start
    print(f"embedded {len(texts):,} jobs in {secs:.1f}s ({len(texts) / secs:.1f} jobs/s)")
    print(f"vector shape: {vectors.shape}, first vector length: {np.linalg.norm(vectors[0]):.4f}")

    # --- Build + save the index ---
    index = vector_index.new_index(vectors.shape[1])
    vector_index.add(index, vectors, df["job_id"])
    vector_index.save(index, out / "jobs.index")
    np.save(out / "embeddings.npy", vectors)  # cached so later experiments don't re-embed

    meta = {"model": MODEL_NAME, "desc_chars": DESC_CHARS, "num_jobs": int(index.ntotal), "dim": int(vectors.shape[1])}
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2))

    size_mb = (out / "jobs.index").stat().st_size / 1e6
    print(f"index holds {index.ntotal:,} vectors, file size {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
