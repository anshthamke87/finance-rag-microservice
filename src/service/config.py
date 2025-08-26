from pathlib import Path
import json, os

def project_root() -> Path:
    # Allow override via env var; default is the repo root (this file's grandparent)
    env = os.environ.get("FIN_RAG_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[2]

def load_best_run():
    root = project_root()
    ptr = root / "reports" / "rag_best_run.json"
    data = json.loads(ptr.read_text())
    # Normalize to absolute paths
    for k in ("manifest","predictions","eval"):
        data[k] = str((root / data[k]).resolve())
    return data

def load_retrieval_selection():
    root = project_root()
    sel = root / "reports" / "retrieval_final_selection.json"
    return json.loads(sel.read_text())
