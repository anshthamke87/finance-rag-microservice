# 🚀 Finance RAG Microservice

**End-to-end Retrieval-Augmented Generation for Finance QA (FiQA)** — from raw data to a production-ready API.  
We: **downloaded & cleaned data → built strong hybrid retrieval (BM25 + dense ANN + CE) → fine-tuned the bi-encoder → generated grounded answers with citations (FLAN-T5)** → packaged everything into a **FastAPI** microservice.


---

## 🔥 Highlights

- **Clean, reproducible pipeline** split into 4 notebooks (NB1–NB4), easy to run in Colab.
- **Strong retriever**: BM25 + HNSW FAISS (inner product) + **cross-encoder reranking**.
- **Fine-tuning**: contrastive training on mined positive/negative pairs from FiQA train/dev.
- **RAG generation**: FLAN‑T5 with **cross-encoder–picked passages** (multiple per doc) and a **token budget** for robust grounding.
- **Citations**: answers include `[DOC:ID]` tags tied to passages so reviewers can verify.
- **Service**: small **FastAPI** app with a single `/answer` endpoint.
- **Artifacts + reports** saved under `artifacts/`, `indices/`, `runs/`, and `reports/`.

---

## 📊 Results (Test)

**Retrieval (hybrid + CE, fine‑tuned)**  
- nDCG@10: **0.3794**  
- MRR@10: **0.4526**  
- Recall@100: **0.6247**

**RAG (CE‑picked passages, tuned prompt)**  
- n_predictions: **648**  
- Support@12: **0.6852**  
- EvidenceRecall@12: **0.4771**  
- ContextContainment: **0.9139**

> _Support@k:_ at least one of the top‑k retrieved docs is truly relevant.  
> _EvidenceRecall@k:_ share of all relevant docs that appear in top‑k.  
> _ContextContainment:_ % of answer tokens found in the provided passages.

---

## 🧠 What we built (end‑to‑end)

### NB1 — Data & Baseline
- 📥 Downloaded **FiQA** (BeIR) corpus + queries + qrels; saved **train/dev/test** splits to Drive.
- 🧩 Created **passages** via chunking; saved as `artifacts/passages.parquet`.
- 🔎 Baseline **BM25** (Rank‑BM25) over passages; reported nDCG/MRR/Recall; stored report JSON.

### NB2 — Hybrid Retrieval + Tuning + Fine‑tune
- 🧮 Built **dense ANN** with **FAISS HNSW (inner product)** on MiniLM embeddings.
- 🔀 **Fusion** of BM25 + ANN via min‑max normalization (α mixing).
- 🤝 **Cross‑encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top‑N candidates.
- ⚙️ **Hyperparam grid** (α, efSearch, CE budget) → dev leaderboard → pick best.
- 🎯 **Fine‑tuned** the bi‑encoder using mined positive/negative pairs (CE‑guided), with in‑notebook checkpoints.
- 🧾 Saved: FAISS index, embeddings, tuned config (`retrieval_final_selection.json`), and final IR metrics.

### NB3 — RAG Generation + Eval
- 📚 **Context selection**: For each top doc, **cross‑encoder scores multiple passages**, keeps the best 2–3 per doc, then globally picks top passages under a **token budget** (e.g., 1,400 tokens). This maximizes the chance the **answer‑bearing** sentence is present.
- 🧾 **Prompting** (FLAN‑T5): concise 1–3 sentence answers; optional **general‑knowledge fallback** labeled with “Based on general knowledge,” to reduce abstains while keeping transparency.
- ✍️ **Decoding**: beam search, `min_new_tokens` to avoid 1‑word replies, light `length_penalty`.
- ✅ **Evaluation**: Support@k, EvidenceRecall@k, ContextContainment, plus quick abstain/length stats.
- 📦 Saved predictions (`runs/*.jsonl`), evals (`reports/*.json`), and a **run manifest** + **best‑run pointer**.

### NB4 — Microservice Packaging (this notebook)
- 📦 Created **FastAPI** app: loads retriever + generator and exposes `POST /answer`.
- 🔧 Tiny **run scripts** and a **HOWTO** for local use.
- 📝 Auto‑generated this **README.md** from your saved metrics.
- (Optional) Smoke tests in Colab using `fastapi.testclient`.

---

## 🧩 Design Choices (and why)

- **BM25 + Dense + CE**: BM25 excels at lexicon overlap; dense covers synonyms/semantics; CE provides precise pairwise scoring on top of candidates. Together, these deliver high recall **and** precision.
- **HNSW + inner product**: scalable ANN for query‑time speed, with normalized vectors for cosine‑like similarity.
- **CE‑picked passages (multiple per doc)**: Avoids the “best single passage per doc” trap; we pick **several** likely answer‑bearing snippets then pack to a budget → lower abstain rate, better grounding.
- **Prompt with labeled fallback**: When context is thin, allowing a **clearly marked** general‑knowledge note improves utility while preserving transparency.
- **Deterministic decoding**: `num_beams=4`, `do_sample=False`, `min_new_tokens` keeps answers consistent and non‑trivial.

---

## 🧪 Reproducibility

- Random seeds set for NumPy / Torch where feasible.
- All major artifacts and configs saved under the repo (`artifacts/`, `indices/`, `reports/`, `runs/`).
- Final tuned retrieval config lives in: `reports/retrieval_final_selection.json`  
- Best RAG run pointer in: `reports/rag_best_run.json`

> If GitHub size is an issue, use Git LFS for `indices/` and `artifacts/`, or regenerate via notebooks.

---

## 📁 Repo Layout

```
.
├── src/service/               # FastAPI app + retriever/generator wrappers
├── scripts/                   # serve + curl examples
├── notebooks/                 # HOWTO + (run NB1..NB3 in Colab)
├── reports/                   # JSON metrics (IR + RAG) + manifests
├── runs/                      # Predictions (JSONL)
├── artifacts/                 # passages.parquet, embeddings, fine-tuned model
├── indices/faiss_hnsw/        # FAISS HNSW index + mapping
└── eval/qrels/                # FiQA qrels & queries (train/dev/test)
```

---

## ▶️ Run the API locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/serve.sh
# new terminal:
./scripts/curl_example.sh
```

> The service loads models at runtime. ✅ GPU recommended; CPU works but slower.

---

## 🔗 API

`POST /answer`

**Request**:
```json
{
  "question": "How do I transfer a 401k after closing a business?",
  "top_k": 12,
  "allow_external_knowledge": true
}
```

**Response**:
```json
{
  "question": "...",
  "answer": "...",
  "citations": [{"doc_id":"...", "passage_idx":123}, ...],
  "used_k": 12
}
```

---

## 📚 Credits & Licenses

- Dataset: **FiQA** (BeIR); academic use — please respect original licenses.
- Models: `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `google/flan-t5-large`.
- Libraries: FAISS, FastAPI, Transformers, Sentence-Transformers, Rank-BM25.

---

_Questions or feedback? Happy to iterate!_ ✨
