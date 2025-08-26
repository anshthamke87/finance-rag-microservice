from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from .config import project_root, load_retrieval_selection
from .retriever import HybridRetriever
from .generator import Generator

app = FastAPI(title="Finance RAG Microservice", version="1.0.0")

ROOT = project_root()
CFG  = load_retrieval_selection()
RET  = HybridRetriever(ROOT, CFG)
GEN  = Generator("google/flan-t5-large", min_new_tokens=24, max_new_tokens=96, num_beams=4)

class QueryIn(BaseModel):
    question: str
    top_k: int = 12
    allow_external_knowledge: bool = True

class AnswerOut(BaseModel):
    question: str
    answer: str
    citations: list
    used_k: int

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/answer", response_model=AnswerOut)
def answer(inp: QueryIn):
    bundle = RET.retrieve(inp.question, k_docs=inp.top_k)
    # build context: take all returned passages (already best per doc)
    context_parts, cites = [], []
    for b in bundle:
        snippet = b["passage"].strip().replace("\n", " ")
        context_parts.append(f"{snippet} [DOC:{b['doc_id']}]")
        cites.append({"doc_id": b["doc_id"], "passage_idx": b["passage_idx"]})
    context = "\n".join(context_parts)

    prompt = GEN.make_prompt(context, inp.question, allow_external_knowledge=inp.allow_external_knowledge)
    ans = GEN.generate(prompt)
    return AnswerOut(question=inp.question, answer=ans, citations=cites, used_k=len(bundle))
