#!/usr/bin/env bash
export FIN_RAG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
uvicorn src.service.app:app --host 0.0.0.0 --port 8000
