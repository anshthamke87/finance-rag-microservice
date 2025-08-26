from __future__ import annotations
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-large",
                 min_new_tokens: int = 24, max_new_tokens: int = 96, num_beams: int = 4):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.gen = pipeline("text2text-generation", model=self.model, tokenizer=self.tok)
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def make_prompt(self, context: str, question: str, allow_external_knowledge: bool = True) -> str:
        if not allow_external_knowledge:
            return (
                "You are a finance assistant. Use ONLY the context to answer in 1–3 sentences. "
                "Include bracketed citations like [DOC:ID] next to claims.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            return (
                "You are a finance assistant. Prefer the context. If the context is insufficient, you may use general knowledge, "
                "and start that part with: 'Based on general knowledge,'. If context contradicts general knowledge, prefer the context. "
                "Include [DOC:ID] citations when you use the context. Answer in 1–3 sentences.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )

    def generate(self, prompt: str) -> str:
        out = self.gen(
            prompt,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=False,
            num_beams=self.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.05
        )[0]["generated_text"]
        return out
