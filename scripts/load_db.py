"""
Step 3: load the 5,000 cleaned jobs (plus their companies and categories) into SQLite.

Usage (from the repo root):
    python -m scripts.load_db
"""
import argparse
from pathlib import Path

import pandas as pd

from app.db import DB_PATH, connect

SCHEMA = Path("app/schema.sql")


def read(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read only the columns we need, with a clear error if the file's columns differ from what we expect."""
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in cols if c not in header]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}. Columns found: {list(header)}")
    return pd.read_csv(path, usecols=cols)


def rows(df: pd.DataFrame) -> list[tuple]:
    """DataFrame -> list of plain-Python tuples. sqlite3 can't store numpy types or pandas' NaN/NA, so convert to int/float/str/None."""
    out = []
    for r in df.itertuples(index=False):
        out.append(tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/jobs_clean.csv")
    p.add_argument("--raw", default="data/raw")
    args = p.parse_args()
    raw = Path(args.raw)

    # ---------- Jobs ----------
    jobs = pd.read_csv(args.csv)
    # Step 1 didn't keep company_id, so fetch it from the raw file by job_id (same 5,000 jobs, no re-sampling).
    ids = read(raw / "postings.csv", ["job_id", "company_id"])
    jobs = jobs.merge(ids, on="job_id", how="left")
    jobs["company_id"] = jobs["company_id"].astype("Int64")  # nullable integer: keeps missing ids as NA, not 1.2e+07 floats
    jobs["listed_at"] = pd.to_datetime(jobs["listed_time"], unit="ms").dt.strftime("%Y-%m-%dT%H:%M:%S")  # epoch ms -> ISO

    # ---------- Companies (only the ones our jobs reference) ----------
    comp = read(raw / "companies" / "companies.csv", ["company_id", "name", "city", "state", "country", "url"])
    comp["company_id"] = comp["company_id"].astype("Int64")
    used = jobs[["company_id", "company_name"]].dropna(subset=["company_id"]).drop_duplicates("company_id")
    companies = used.merge(comp, on="company_id", how="left")
    # If companies.csv lacks a company, fall back to the name on the posting so every referenced company exists (FK holds).
    companies["name"] = companies["name"].fillna(companies["company_name"]).fillna("Unknown")
    companies = companies[["company_id", "name", "city", "state", "country", "url"]]

    # ---------- Categories ----------
    cats = read(raw / "mappings" / "skills.csv", ["skill_abr", "skill_name"]).drop_duplicates("skill_abr")
    job_cats = read(raw / "jobs" / "job_skills.csv", ["job_id", "skill_abr"]).drop_duplicates()
    job_cats = job_cats[job_cats["job_id"].isin(jobs["job_id"])]
    unknown = ~job_cats["skill_abr"].isin(cats["skill_abr"])
    if unknown.any():
        print(f"dropping {unknown.sum()} job-category links with codes not in skills.csv")
    job_cats = job_cats[~unknown]

    # ---------- Write: start from an empty database every time, so the script is repeatable ----------
    DB_PATH.unlink(missing_ok=True)
    conn = connect()
    conn.executescript(SCHEMA.read_text())
    with conn:  # one transaction: either everything is inserted or nothing is
        conn.executemany("INSERT INTO companies VALUES (?,?,?,?,?,?)", rows(companies))
        job_cols = ["job_id", "company_id", "title", "description", "location", "formatted_experience_level",
                    "formatted_work_type", "remote_allowed", "min_salary", "max_salary", "pay_period",
                    "listed_at", "job_posting_url"]
        conn.executemany(
            """INSERT INTO jobs (job_id, company_id, title, description, location, experience_level, work_type,
                                 remote_allowed, min_salary, max_salary, pay_period, listed_at, url)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows(jobs[job_cols]),
        )
        conn.executemany("INSERT INTO categories VALUES (?,?)", rows(cats))
        conn.executemany("INSERT INTO job_categories VALUES (?,?)", rows(job_cats))

    # ---------- Verify ----------
    q = lambda sql: conn.execute(sql).fetchall()
    for t in ["companies", "jobs", "categories", "job_categories"]:
        print(f"{t:15s} {q(f'SELECT COUNT(*) FROM {t}')[0][0]:>6,} rows")
    print(f"jobs with no company_id:  {q('SELECT COUNT(*) FROM jobs WHERE company_id IS NULL')[0][0]:,}")
    print(f"jobs with no category:    {q('SELECT COUNT(*) FROM jobs WHERE job_id NOT IN (SELECT job_id FROM job_categories)')[0][0]:,}")
    print(f"foreign key violations:   {len(q('PRAGMA foreign_key_check'))}")

    print("\nJobs per category (top 10):")
    for r in q("""SELECT c.name, COUNT(*) AS n FROM job_categories jc
                  JOIN categories c ON c.category_code = jc.category_code
                  GROUP BY c.name ORDER BY n DESC LIMIT 10"""):
        print(f"  {r['n']:>5,}  {r['name']}")

    print("\nQuery plan for 'all jobs in category IT' (should mention idx_job_categories_by_cat):")
    for r in q("EXPLAIN QUERY PLAN SELECT job_id FROM job_categories WHERE category_code = 'IT'"):
        print("  ", r["detail"])

    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"\nsaved {DB_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
