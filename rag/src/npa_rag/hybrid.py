"""Hybrid retrieval: BM25 + dense vectors, fused with reciprocal rank fusion, optionally reranked.

    RRF(d) = sum over retrievers r of 1 / (RRF_K + rank_r(d)),  rank starting at 1, top CANDIDATES per retriever

RRF uses ranks only, so the incomparable BM25 and cosine scales never have to be calibrated against each other.
The cross-encoder (if enabled) rescores the top RERANK_DEPTH fused chunks against the query text.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from forte_kb.store import Space

from . import config
from .corpus import Chunk, load_chunks
from .sparse import Sparse


@dataclass
class Hit:
    chunk: Chunk
    rrf: float
    ranks: dict[str, int]                 # rank in each retriever's top list (absent = not in it)
    rerank: float | None = None

    def as_dict(self) -> dict:
        c = self.chunk
        return {"id": c.id, "citation": c.citation, "act_id": c.act_id, "act_requisites": c.act_requisites,
                "label": c.label, "part": c.part, "status": c.status, "kind": c.kind, "source": c.source,
                "text": c.text, "rrf": round(self.rrf, 5), "ranks": self.ranks,
                "rerank": None if self.rerank is None else round(float(self.rerank), 4)}


@dataclass
class Result:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


def top_ranks(scores: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
    """Indices of the n best-scoring rows among mask, best first; rows scoring 0 or less are dropped."""
    s = np.where(mask, scores, -np.inf)
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return []
    top = np.argpartition(-s, n - 1)[:n]
    top = top[np.argsort(-s[top], kind="stable")]
    return [int(i) for i in top if s[i] > 0]


def rrf_fuse(rank_lists: dict[str, list[int]], k: int = config.RRF_K) -> list[tuple[int, float, dict[str, int]]]:
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, lst in rank_lists.items():
        for r, i in enumerate(lst, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
            ranks.setdefault(i, {})[name] = r
    order = sorted(fused, key=lambda i: (-fused[i], min(ranks[i].values())))
    return [(i, fused[i], ranks[i]) for i in order]


class Retriever:
    def __init__(self, store: Path, embedder=None, reranker=None):
        self.chunks = load_chunks(store / "chunks.jsonl")
        self.sparse = Sparse.load(store / "bm25")
        self.dense = Space.load(store, "dense") if (store / "dense.npy").exists() else None
        ids = [c.id for c in self.chunks]
        if self.sparse.ids != ids or (self.dense is not None and [m["id"] for m in self.dense.meta] != ids):
            raise ValueError("chunks, bm25 and dense indexes are out of sync: rebuild with `rag build`")
        self.kinds = np.array([c.kind for c in self.chunks])
        self.embedder = embedder
        self.reranker = reranker

    def mask(self, kinds=config.SEARCH_KINDS, acts: set[str] | None = None) -> np.ndarray:
        m = np.isin(self.kinds, list(kinds))
        if acts:
            m &= np.isin(np.array([c.act_id for c in self.chunks]), list(acts))
        return m

    def query_vector(self, query: str) -> np.ndarray:
        from .dense import embed_query
        if self.embedder is None:
            raise RuntimeError("dense search needs an embedder")
        return embed_query(self.embedder, query)

    def candidates(self, query: str, mask: np.ndarray, methods=("bm25", "dense"), n: int = config.CANDIDATES,
                   query_vec: np.ndarray | None = None, timings: dict | None = None) -> dict[str, list[int]]:
        timings = {} if timings is None else timings
        lists = {}
        if "bm25" in methods:
            t = time.perf_counter()
            lists["bm25"] = top_ranks(self.sparse.scores(query), mask, n)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if "dense" in methods:
            t = time.perf_counter()
            q = query_vec if query_vec is not None else self.query_vector(query)
            timings["embed_query"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            sims = self.dense.vectors @ (q / (np.linalg.norm(q) or 1.0))
            lists["dense"] = top_ranks(sims + 1.0, mask, n)  # +1 keeps every cosine positive for top_ranks
            timings["dense"] = (time.perf_counter() - t) * 1000
        return lists

    def search(self, query: str, k: int = 10, kinds=config.SEARCH_KINDS, acts: set[str] | None = None,
               methods=("bm25", "dense"), rerank: bool = False, depth: int = config.RERANK_DEPTH,
               query_vec: np.ndarray | None = None) -> Result:
        timings: dict[str, float] = {}
        lists = self.candidates(query, self.mask(kinds, acts), methods, query_vec=query_vec, timings=timings)
        fused = rrf_fuse(lists)
        hits = [Hit(self.chunks[i], score, r) for i, score, r in fused[: max(k, depth if rerank else k)]]
        if rerank and hits:
            if self.reranker is None:
                raise RuntimeError("rerank=True needs a reranker")
            t = time.perf_counter()
            scores = self.reranker.score(query, [rerank_text(h.chunk) for h in hits])
            timings["rerank"] = (time.perf_counter() - t) * 1000
            for h, s in zip(hits, scores):
                h.rerank = float(s)
            hits.sort(key=lambda h: -h.rerank)
        return Result(query, hits[:k], timings)


def rerank_text(c: Chunk) -> str:
    head = " / ".join([c.segment[:200], *c.path[-1:]])
    return f"{head}. {c.label}. {c.text}" if c.label else f"{head}. {c.text}"
