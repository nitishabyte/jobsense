"""
All SQLite access goes through here, so the API and scripts share one way of talking to the database.
"""
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "data/jobsense.db"))


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    # check_same_thread=False: FastAPI may open the connection in one worker thread and use it in another.
    # Safe here because each request gets its OWN connection; we never share one between requests.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row             # rows behave like dicts: row["title"]
    conn.execute("PRAGMA foreign_keys = ON")   # SQLite ignores REFERENCES unless this is on, per connection
    return conn


# ---------- Reads ----------

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
               GROUP_CONCAT(cat.name, ', ')        AS categories,
               GROUP_CONCAT(jc.category_code, ',') AS category_codes
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


def all_job_ids(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT job_id FROM jobs")}


def list_categories(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT c.category_code, c.name, COUNT(jc.job_id) AS num_jobs
           FROM categories c LEFT JOIN job_categories jc ON jc.category_code = c.category_code
           GROUP BY c.category_code ORDER BY num_jobs DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def unknown_categories(conn: sqlite3.Connection, codes: list[str]) -> list[str]:
    known = {r[0] for r in conn.execute("SELECT category_code FROM categories")}
    return sorted(set(codes) - known)


# ---------- Writes (the caller wraps them in a transaction: `with conn:`) ----------

def get_or_create_company(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT company_id FROM companies WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO companies (name) VALUES (?)", (name,)).lastrowid


def insert_job(conn: sqlite3.Connection, job: dict) -> int:
    """Insert a job and its categories; returns the new job_id (SQLite picks max(job_id) + 1)."""
    company_id = get_or_create_company(conn, job["company_name"]) if job.get("company_name") else None
    job_id = conn.execute(
        """INSERT INTO jobs (company_id, title, description, location, remote_allowed, url, listed_at)
           VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S', 'now'))""",
        (company_id, job["title"], job["description"], job.get("location"),
         int(job.get("remote_allowed", False)), job.get("url")),
    ).lastrowid
    conn.executemany(
        "INSERT INTO job_categories (job_id, category_code) VALUES (?, ?)",
        [(job_id, code) for code in set(job.get("category_codes", []))],
    )
    return job_id


def delete_job(conn: sqlite3.Connection, job_id: int) -> bool:
    """Delete a job (its category links go too, via ON DELETE CASCADE). Returns False if it didn't exist."""
    return conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,)).rowcount > 0
