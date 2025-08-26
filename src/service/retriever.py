from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

class HybridRetriever:
    def __init__(self, root: Path, final_cfg: dict):
        self.root = Path(root)
        self.cfg = final_cfg

        # Passages & mapping
        self.passages_df = pd.read_parquet(self.root / self.cfg["passages"]["file"])
        self.passages = self.passages_df["passage"].tolist()
        self.passage_docids = self.passages_df["doc_id"].tolist()
        self.DOC2PIs = {}
        for i, did in enumerate(self.passage_docids):
            self.DOC2PIs.setdefault(did, []).append(i)

        # BM25 over passages
        self.bm25 = BM25Okapi([p.split() for p in self.passages],
                              k1=self.cfg["bm25"]["k1"], b=self.cfg["bm25"]["b"])

        # FAISS HNSW index (inner product on normalized vectors)
        self.index = faiss.read_index(str(self.root / self.cfg["ann"]["index_path"]))
        try:
            self.index.hnsw.efSearch = self.cfg["ann"]["efSearch"]
        except Exception:
            pass

        # Query encoder (fine-tuned)
        self.q_encoder = SentenceTransformer(str(self.root / self.cfg["bi_encoder_model_path"]),
                                             device="cuda" if torch.cuda.is_available() else "cpu")

        # Cross-encoder reranker
        ce_name = self.cfg["ce"]["model"]
        self.ce_tokenizer = AutoTokenizer.from_pretrained(ce_name)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(ce_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        ).eval()

    def _bm25_scores(self, q: str):
        return self.bm25.get_scores(q.split())

    def _bm25_doc_aggregate(self, scores, top_passages=2000, top_docs=500):
        k = min(top_passages, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        doc2score, doc_best_pi = {}, {}
        for pi in top_idx:
            did = self.passage_docids[pi]
            s = float(scores[pi])
            if (did not in doc2score) or (s > doc2score[did]):
                doc2score[did] = s
                doc_best_pi[did] = int(pi)
        ranked = sorted(doc2score.items(), key=lambda x: x[1], reverse=True)[:top_docs]
        return ranked, doc_best_pi

    def _ann_search(self, q: str, topk=200) -> Tuple[List[Tuple[int,float]], Dict[str,Tuple[int,float]]]:
        qv = self.q_encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idx = self.index.search(qv, topk)
        idx = idx[0]; scores = scores[0]
        pairs = [(int(i), float(s)) for i, s in zip(idx, scores) if i != -1]
        doc_best = {}
        for pi, s in pairs:
            did = self.passage_docids[pi]
            if (did not in doc_best) or (s > doc_best[did][1]):
                doc_best[did] = (int(pi), float(s))
        return pairs, doc_best

    @staticmethod
    def _fuse_minmax_docs(bm25_ranked, bm25_best_pi, ann_doc_best, alpha=0.4):
        b_dict = {d: s for d, s in bm25_ranked}
        if b_dict:
            b = np.fromiter(b_dict.values(), dtype=np.float32)
            bmin, bmax = float(b.min()), float(b.max())
        else:
            bmin, bmax = 0.0, 1.0
        b_norm = {d: (s - bmin) / (bmax - bmin + 1e-9) for d, s in b_dict.items()}

        a_dict = {d: s for d, (pi, s) in ann_doc_best.items()}
        if a_dict:
            a = np.fromiter(a_dict.values(), dtype=np.float32)
            amin, amax = float(a.min()), float(a.max())
        else:
            amin, amax = 0.0, 1.0
        a_norm = {d: (s - amin) / (amax - amin + 1e-9) for d, s in a_dict.items()}

        docs = set(b_dict) | set(a_dict)
        fused, doc_best_pi = [], {}
        for d in docs:
            bn = b_norm.get(d, 0.0)
            an = a_norm.get(d, 0.0)
            fused.append((d, alpha * bn + (1 - alpha) * an))
            doc_best_pi[d] = ann_doc_best[d][0] if d in ann_doc_best else bm25_best_pi.get(d)
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused, doc_best_pi

    @torch.no_grad()
    def _rerank_with_ce(self, query: str, fused_docs, doc_best_pi, top_for_ce=50, max_len=256):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cand = fused_docs[:top_for_ce]
        ids, pairs = [], []
        for d, _ in cand:
            pi = doc_best_pi.get(d)
            if pi is None: continue
            ids.append(d)
            pairs.append((query, self.passages[pi]))
        if not pairs:
            return [d for d, _ in fused_docs[:100]]
        batch = self.ce_tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        scores = self.ce_model(**batch).logits.squeeze(-1).detach().float().cpu().numpy()
        order = scores.argsort()[::-1]
        return [ids[i] for i in order]

    def retrieve(self, query: str, k_docs: int = 12):
        p_scores = self._bm25_scores(query)
        bm25_docs, bm25_best_pi = self._bm25_doc_aggregate(p_scores, top_passages=2000, top_docs=500)
        _, ann_doc_best = self._ann_search(query, topk=200)
        fused_docs, doc_best_pi = self._fuse_minmax_docs(bm25_docs, bm25_best_pi, ann_doc_best, alpha=self.cfg["fusion"]["alpha"])
        final_docs = self._rerank_with_ce(query, fused_docs, doc_best_pi, top_for_ce=self.cfg["ce"]["top_for_ce"])
        # Return top docs + their best passage indices
        out = []
        for d in final_docs[:k_docs]:
            pi = doc_best_pi.get(d)
            if pi is None:
                continue
            out.append({"doc_id": d, "passage_idx": int(pi), "passage": self.passages[pi]})
        return out
