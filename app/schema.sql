-- JobSense schema. SQLite holds everything ABOUT a job; FAISS holds only its vector.
-- job_id is the shared key between the two.

CREATE TABLE companies (
    company_id  INTEGER PRIMARY KEY,          -- LinkedIn's id, reused so we can join back to the raw data
    name        TEXT    NOT NULL,
    city        TEXT,
    state       TEXT,
    country     TEXT,
    url         TEXT
);

CREATE TABLE jobs (
    job_id            INTEGER PRIMARY KEY,    -- same id that FAISS stores with each vector
    company_id        INTEGER REFERENCES companies(company_id),   -- NULL allowed: some postings have no company id
    title             TEXT    NOT NULL,
    description       TEXT    NOT NULL,
    location          TEXT,
    experience_level  TEXT,                   -- e.g. 'Entry level', 'Mid-Senior level'
    work_type         TEXT,                   -- e.g. 'Full-time', 'Contract'
    remote_allowed    INTEGER NOT NULL DEFAULT 0 CHECK (remote_allowed IN (0, 1)),  -- SQLite has no BOOLEAN
    min_salary        REAL,
    max_salary        REAL,
    pay_period        TEXT,                   -- 'HOURLY', 'YEARLY', ... salaries are only comparable within one period
    listed_at         TEXT,                   -- ISO-8601 text: SQLite has no DATE type, and ISO strings sort correctly
    url               TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),  -- when WE inserted it (useful once jobs arrive via the API)
    CHECK (min_salary IS NULL OR max_salary IS NULL OR min_salary <= max_salary)
);

-- LinkedIn's broad job categories ("IT" = Information Technology, "SALE" = Sales, ...).
-- The dataset calls these "skills", but they are job functions, so we name them honestly.
CREATE TABLE categories (
    category_code  TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE
);

-- Many-to-many: one job has several categories, one category has many jobs.
CREATE TABLE job_categories (
    job_id         INTEGER NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    category_code  TEXT    NOT NULL REFERENCES categories(category_code),
    PRIMARY KEY (job_id, category_code)       -- no duplicate pairs; also serves "categories of job X" lookups
);

-- Indexes only for lookups we actually do:
CREATE INDEX idx_jobs_company           ON jobs(company_id);                  -- "all jobs at company X", join to companies
CREATE INDEX idx_job_categories_by_cat  ON job_categories(category_code, job_id);  -- "all jobs in category X"
