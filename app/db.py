"""
All SQLite access goes through here, so the API and scripts share one way of talking to the database.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/jobsense.db")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row             # rows behave like dicts: row["title"]
    conn.execute("PRAGMA foreign_keys = ON")   # SQLite ignores REFERENCES unless this is on, per connection
    return conn


def fetch_jobs(conn: sqlite3.Connection, job_ids: list[int]) -> dict[int, dict]:
    """Fetch full details for the job_ids FAISS returned, keyed by job_id (FAISS decides the order, not SQL)."""
    if not job_ids:
        return {}
    placeholders = ",".join("?" * len(job_ids))  # "?,?,?" -> values are passed separately: no SQL injection
    rows = conn.execute(
        f"""
        SELECT j.job_id, j.title, c.name AS company, j.location, j.description,
               j.experience_level, j.work_type, j.remote_allowed,
               j.min_salary, j.max_salary, j.pay_period, j.listed_at, j.url,
               GROUP_CONCAT(cat.name, ', ') AS categories
        FROM jobs j
        LEFT JOIN companies c        ON c.company_id = j.company_id
        LEFT JOIN job_categories jc  ON jc.job_id = j.job_id
        LEFT JOIN categories cat     ON cat.category_code = jc.category_code
        WHERE j.job_id IN ({placeholders})
        GROUP BY j.job_id
        """,
        job_ids,
    ).fetchall()
    return {r["job_id"]: dict(r) for r in rows}
