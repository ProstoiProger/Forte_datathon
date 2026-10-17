"""On-disk vector spaces with exact cosine search.

A space is two files: <name>.npy (float32, L2-normalized rows) and <name>.jsonl (one metadata object per row,
each with a unique "id"). These files are the source of truth; any external vector database is loaded from them.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


class Space:
    def __init__(self, name: str, vectors: np.ndarray, meta: list[dict]):
        if len(vectors) != len(meta):
            raise ValueError(f"{name}: {len(vectors)} vectors but {len(meta)} metadata rows")
        self.name = name
        self.vectors = vectors.astype(np.float32, copy=False)
        self.meta = meta
        self._index = {m["id"]: i for i, m in enumerate(meta)}
        if len(self._index) != len(meta):
            raise ValueError(f"{name}: duplicate ids")

    def __len__(self) -> int:
        return len(self.meta)

    def vector(self, id_: str) -> np.ndarray:
        return self.vectors[self._index[id_]]

    def search(self, query: np.ndarray, k: int = 10, where: Callable[[dict], bool] | None = None) -> list[dict]:
        """Top-k rows by cosine similarity; `where` filters rows by metadata."""
        if not len(self):
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        if where is not None:
            keep = np.fromiter((where(m) for m in self.meta), dtype=bool, count=len(self.meta))
            scores = np.where(keep, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-scores[top])]
        return [{**self.meta[i], "score": float(scores[i])} for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{self.name}.npy", self.vectors)
        with (directory / f"{self.name}.jsonl").open("w", encoding="utf-8") as fh:
            for m in self.meta:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path, name: str) -> "Space":
        vectors = np.load(directory / f"{name}.npy")
        with (directory / f"{name}.jsonl").open(encoding="utf-8") as fh:
            meta = [json.loads(line) for line in fh if line.strip()]
        return cls(name, vectors, meta)
