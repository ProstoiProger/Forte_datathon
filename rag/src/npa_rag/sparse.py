"""BM25 over chunk texts: Russian Snowball stemming and stopwords, the same analysis as Elasticsearch's
"russian" analyzer, but in-process (no server to keep alive)."""

import json
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

_STEMMER = Stemmer.Stemmer("russian")


def _normalize(texts: list[str]) -> list[str]:
    return [t.replace("ё", "е").replace("Ё", "Е") for t in texts]


def tokenize(texts: list[str], show_progress: bool = False) -> list[list[str]]:
    return bm25s.tokenize(_normalize(texts), stopwords="ru", stemmer=_STEMMER, return_ids=False,
                          show_progress=show_progress)


class Sparse:
    def __init__(self, retriever: bm25s.BM25, ids: list[str]):
        self.retriever = retriever
        self.ids = ids

    @classmethod
    def build(cls, texts: list[str], ids: list[str]) -> "Sparse":
        retriever = bm25s.BM25(k1=1.2, b=0.75)
        retriever.index(tokenize(texts, show_progress=True), show_progress=False)
        return cls(retriever, ids)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory))
        (directory / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Sparse":
        retriever = bm25s.BM25.load(str(directory))
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return cls(retriever, ids)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every chunk for one query (0 where no query term occurs)."""
        tokens = tokenize([query])[0]
        tokens = [t for t in tokens if t in self.retriever.vocab_dict]
        if not tokens:
            return np.zeros(len(self.ids), dtype=np.float32)
        return np.asarray(self.retriever.get_scores(tokens), dtype=np.float32)
