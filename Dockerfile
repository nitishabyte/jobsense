# JobSense API image.
# Build:  docker build -t jobsense .
# Run:    docker run -p 8000:8000 -e ADMIN_API_KEY=... jobsense

# Same Python version as the Codespace we developed and tested in (dev/prod parity). "slim" = small Debian base.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Render's free tier has 0.1 CPU: extra threads only add overhead and memory.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    # Fewer memory arenas in glibc's allocator = lower memory use for a multi-threaded Python server.
    MALLOC_ARENA_MAX=2 \
    INDEX_PATH=/app/data/jobs.index \
    DB_PATH=/app/data/jobsense.db

WORKDIR /app

# Run as a normal user, not root: if the app were ever compromised, the attacker gets fewer permissions.
RUN useradd --create-home appuser && mkdir -p /app/data && chown appuser /app/data

# Install dependencies BEFORE copying code: Docker caches this slow layer, so code changes rebuild in seconds.
# CPU-only PyTorch first - the default PyPI build bundles GPU (CUDA) libraries we can't use and is several GB.
COPY requirements-deploy.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-deploy.txt

USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# Download the embedding model at BUILD time, so a cold start never waits for (or depends on) the Hugging Face Hub.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY --chown=appuser app/ app/
COPY --chown=appuser data/jobs.index data/jobsense.db data/

EXPOSE 8000
# Render tells the app which port to use via $PORT; locally we fall back to 8000. One worker = one copy of the model in RAM.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
