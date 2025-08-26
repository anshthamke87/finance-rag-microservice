# HOWTO (quick)
- Start the API locally:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ./scripts/serve.sh
  ```
- Test it:
  ```bash
  ./scripts/curl_example.sh
  ```
- Tip: set `FIN_RAG_ROOT` to point at the repo if you start uvicorn another way:
  ```bash
  FIN_RAG_ROOT=$(pwd) uvicorn src.service.app:app --reload
  ```
