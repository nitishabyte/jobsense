"""
JobSense API.

Run:   uvicorn app.main:app --reload
Docs:  http://localhost:8000/docs
"""
import os
import secrets
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app import db, vector_index
from app.embedding import MODEL_NAME, encode, get_model, job_to_text

INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/jobs.index"))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")  # not set -> write endpoints are switched off
MAX_TOP_K = 50
FILTER_CANDIDATES = 200  # with a category filter, take this many from FAISS, then filter down to top_k

index_lock = threading.Lock()  # FAISS is not safe to search and modify at the same time
state: dict = {}               # holds the loaded index for the life of the server


# ---------- Startup ----------

def reconcile(index, conn) -> tuple[int, int]:
    """SQLite is the source of truth; FAISS is derived from it. Make FAISS match SQLite."""
    db_ids, index_ids = db.all_job_ids(conn), vector_index.ids_in(index)
    missing, extra = sorted(db_ids - index_ids), sorted(index_ids - db_ids)
    if missing:  # in the database but not searchable -> embed and add
        jobs = db.fetch_jobs(conn, missing)
        vector_index.add(index, encode([job_to_text(jobs[i]["title"], jobs[i]["description"]) for i in missing]), missing)
    if extra:    # searchable but deleted from the database -> remove
        vector_index.remove(index, extra)
    return len(missing), len(extra)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # load the model once, now, instead of on the first request
    index = vector_index.load(INDEX_PATH)
    conn = db.connect()
    try:
        added, removed = reconcile(index, conn)
    finally:
        conn.close()
    if added or removed:
        vector_index.save(index, INDEX_PATH)
    print(f"JobSense ready: {index.ntotal} jobs indexed (reconciled +{added} / -{removed})")
    state["index"] = index
    yield  # the server runs while we're paused here


app = FastAPI(title="JobSense", description="Semantic job search", version="1.0", lifespan=lifespan)


# ---------- Dependencies: small functions FastAPI runs before an endpoint ----------

def get_conn():
    """One SQLite connection per request, always closed afterwards."""
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


def require_admin(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Writes need the X-API-Key header, so strangers can't add or delete jobs on the live demo."""
    if not ADMIN_API_KEY:
        raise HTTPException(503, "Write endpoints are disabled: ADMIN_API_KEY is not set on the server")
    if not x_api_key or not secrets.compare_digest(x_api_key, ADMIN_API_KEY):  # constant-time comparison
        raise HTTPException(401, "Missing or wrong X-API-Key header")


# ---------- Request / response shapes (FastAPI validates and documents these automatically) ----------

class JobResult(BaseModel):
    rank: int
    job_id: int
    score: float
    title: str
    company: str | None
    location: str | None
    categories: str | None
    remote_allowed: bool
    url: str | None
    snippet: str


class SearchResponse(BaseModel):
    query: str
    category: str | None
    took_ms: float
    results: list[JobResult]


class JobIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=50, max_length=20_000)
    company_name: str | None = Field(default=None, max_length=200)
    location: str | None = Field(default=None, max_length=200)
    category_codes: list[str] = []
    remote_allowed: bool = False
    url: str | None = Field(default=None, max_length=500)

    @field_validator("title", "description")
    @classmethod
    def collapse_whitespace(cls, v: str) -> str:
        return " ".join(v.split())  # same cleaning as Step 1, so new jobs are embedded like the old ones


# ---------- Endpoints ----------
# Plain `def` (not `async def`): embedding and FAISS are CPU work that would block an async event loop.
# FastAPI runs `def` endpoints in a thread pool instead.

@app.get("/health")
def health():
    return {"status": "ok", "jobs_indexed": state["index"].ntotal, "model": MODEL_NAME}


@app.get("/search", response_model=SearchResponse)
def search(
    conn: Conn,
    query: Annotated[str, Query(min_length=2, max_length=500, description="What you're looking for, in plain words")],
    top_k: Annotated[int, Query(ge=1, le=MAX_TOP_K)] = 5,
    category: Annotated[str | None, Query(description="Optional category code, e.g. IT (see /categories)")] = None,
):
    start = time.perf_counter()
    if category and db.unknown_categories(conn, [category]):
        raise HTTPException(400, f"Unknown category '{category}'. See /categories.")

    query_vector = encode([query])
    with index_lock:
        hits = vector_index.search(state["index"], query_vector, FILTER_CANDIDATES if category else top_k)
    jobs = db.fetch_jobs(conn, [job_id for job_id, _ in hits])

    results = []
    for job_id, score in hits:  # FAISS order = best first
        job = jobs.get(job_id)
        if job is None:
            continue
        if category and category not in (job["category_codes"] or "").split(","):
            continue
        results.append(JobResult(
            rank=len(results) + 1, job_id=job_id, score=round(score, 4),
            title=job["title"], company=job["company"], location=job["location"],
            categories=job["categories"], remote_allowed=bool(job["remote_allowed"]),
            url=job["url"], snippet=job["description"][:300],
        ))
        if len(results) == top_k:
            break

    took_ms = round((time.perf_counter() - start) * 1000, 1)
    return SearchResponse(query=query, category=category, took_ms=took_ms, results=results)


@app.get("/categories")
def categories(conn: Conn):
    return db.list_categories(conn)


@app.get("/jobs/{job_id}")
def get_job(job_id: int, conn: Conn):
    job = db.fetch_jobs(conn, [job_id]).get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/jobs", status_code=201, dependencies=[Depends(require_admin)])
def add_job(job: JobIn, conn: Conn):
    """Add a posting and make it searchable immediately - no index rebuild."""
    start = time.perf_counter()
    unknown = db.unknown_categories(conn, job.category_codes)
    if unknown:
        raise HTTPException(400, f"Unknown category codes: {unknown}. See /categories.")

    vector = encode([job_to_text(job.title, job.description)])  # 1. embed first: slowest step, nothing saved yet
    with conn:                                                   # 2. one transaction: job + company + categories
        job_id = db.insert_job(conn, job.model_dump())
    with index_lock:                                             # 3. make it searchable, then persist the index
        vector_index.add(state["index"], vector, [job_id])
        vector_index.save(state["index"], INDEX_PATH)
    # If step 3 ever fails, the job is safe in SQLite and reconcile() adds it at the next startup.

    took_ms = round((time.perf_counter() - start) * 1000, 1)
    return {"job_id": job_id, "jobs_indexed": state["index"].ntotal, "took_ms": took_ms}


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_admin)])
def delete_job(job_id: int, conn: Conn):
    with conn:
        if not db.delete_job(conn, job_id):
            raise HTTPException(404, "Job not found")
    with index_lock:
        vector_index.remove(state["index"], [job_id])
        vector_index.save(state["index"], INDEX_PATH)
    return {"deleted": job_id, "jobs_indexed": state["index"].ntotal}
