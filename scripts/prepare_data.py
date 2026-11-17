"""
Step 1: Clean the raw LinkedIn postings CSV and save a smaller, tidy sample.

Usage:
    python scripts/prepare_data.py --raw data/raw/postings.csv --out data/jobs_clean.csv --n 5000
"""
import argparse
import re

import pandas as pd

# The only columns we need. Everything else in the raw file is ignored.
KEEP = [
    "job_id",
    "title",
    "company_name",
    "location",
    "description",
    "formatted_experience_level",
    "formatted_work_type",
    "remote_allowed",
    "min_salary",
    "max_salary",
    "pay_period",
    "listed_time",
    "job_posting_url",
]


def clean_text(s: str) -> str:
    """Collapse all runs of whitespace (newlines, tabs, double spaces) into one space."""
    return re.sub(r"\s+", " ", str(s)).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="path to the raw postings.csv")
    p.add_argument("--out", required=True, help="where to write the cleaned CSV")
    p.add_argument("--n", type=int, default=5000, help="how many postings to keep")
    p.add_argument("--seed", type=int, default=42, help="random seed, so the sample is reproducible")
    p.add_argument("--min-desc-chars", type=int, default=200)
    args = p.parse_args()

    df = pd.read_csv(args.raw)
    print(f"raw rows: {len(df):,}")

    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"These expected columns are missing: {missing}\nColumns found: {list(df.columns)}")
    df = df[KEEP]

    # 1. Drop rows with no title or no description: nothing to search on.
    df = df.dropna(subset=["title", "description"])
    print(f"after dropping empty title/description: {len(df):,}")

    # 2. Normalise whitespace in the text fields.
    for col in ["title", "company_name", "location", "description"]:
        df[col] = df[col].fillna("").map(clean_text)

    # 3. Drop very short descriptions ("See website", "Apply now") - they carry no meaning to embed.
    df = df[df["description"].str.len() >= args.min_desc_chars]
    print(f"after dropping descriptions < {args.min_desc_chars} chars: {len(df):,}")

    # 4. Remove duplicates: the same job is often re-posted with a new job_id.
    df = df.drop_duplicates(subset=["job_id"])
    df = df.drop_duplicates(subset=["title", "company_name", "description"])
    print(f"after de-duplication: {len(df):,}")

    # 5. Take a reproducible random sample so embedding fits on a laptop / free tier.
    if len(df) > args.n:
        df = df.sample(n=args.n, random_state=args.seed)
    df = df.sort_values("job_id").reset_index(drop=True)

    # 6. remote_allowed is 1.0 or empty in the raw data -> make it a clean 0/1 integer.
    df["remote_allowed"] = df["remote_allowed"].fillna(0).astype(int)

    df.to_csv(args.out, index=False)
    print(f"saved {len(df):,} rows to {args.out}")
    print("\nTop 15 titles in the sample:")
    print(df["title"].value_counts().head(15).to_string())
    print(f"\nDescription length (chars): median={df['description'].str.len().median():.0f}, "
          f"max={df['description'].str.len().max():,}")


if __name__ == "__main__":
    main()
