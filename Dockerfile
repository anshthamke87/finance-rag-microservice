# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (curl for healthcheck; git optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
# (expects requirements.txt at build context root)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy minimal code needed to run the API (mount the full repo at runtime)
COPY src /app/src
COPY scripts /app/scripts

# Default project root inside the container (can override at runtime)
ENV FIN_RAG_ROOT=/app
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# Optional: healthcheck pings /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fs http://localhost:8000/health || exit 1

# Start FastAPI app
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
