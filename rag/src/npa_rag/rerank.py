"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3, local files only)."""

import gc

import numpy as np


class Reranker:
    def __init__(self, model: str, device: str | None = None, max_length: int = 1024):
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def load(self) -> "Reranker":
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, dtype=dtype, local_files_only=True).to(device).eval()
        return self

    def unload(self) -> None:
        import torch
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> np.ndarray:
        """Relevance in 0..1 for each (query, passage) pair."""
        import torch
        self.load()
        out = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation="only_second",
                                 max_length=self.max_length, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.view(-1).float()
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
