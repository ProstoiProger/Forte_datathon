"""HTTP search service: `uv run rag serve`.

The retriever loads once, in a background thread, so the port answers /health at once ("ready": false while the
embedder loads). One lock serializes searches: the embedder and the reranker are not safe for concurrent calls, and
BM25 plus the dense dot product take a few milliseconds next to the query embedding.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config

log = logging.getLogger(__name__)

WARMUP_QUERY = "кассовый работник покидает рабочее место, оставив наличные деньги на столе"

Kind = Literal["base", "repealed"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(3, ge=1, le=10)
    kinds: list[Kind] = Field(default_factory=lambda: ["base"], min_length=1)
    rerank: bool = False

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query is blank")
        return value


class HitOut(BaseModel):
    id: str
    act_id: str
    citation: str
    act_title: str
    act_requisites: str
    label: str
    part: str
    text: str
    url: str
    status: str
    kind: str
    rrf: float
    ranks: dict[str, int]
    rerank: float | None


class SearchResponse(BaseModel):
    query: str
    hits: list[HitOut]
    timings_ms: dict[str, float]


class Health(BaseModel):
    ready: bool
    chunks: int
    device: str
    rerank: bool
    error: str | None = None


class State:
    def __init__(self, device: str, rerank: bool) -> None:
        self.device = device
        self.rerank = rerank
        self.retriever = None
        self.error: str | None = None
        self.lock = threading.Lock()


def load_retriever(store: Path, device: str, rerank: bool):
    from forte_kb.embedder import Embedder

    from .hybrid import Retriever
    reranker = None
    if rerank:
        from .rerank import Reranker
        reranker = Reranker(config.RERANK_MODEL, device=device)
    return Retriever(store, embedder=Embedder(config.EMBED_MODEL, config.EMBED_DIM, device=device), reranker=reranker)


def warm_up(retriever, rerank: bool) -> None:
    """Loads the models lazily held by the retriever; depth=2 keeps a CPU rerank short."""
    retriever.search(WARMUP_QUERY, k=1, rerank=rerank, depth=2)


def _start(state: State, store: Path) -> None:
    try:
        t = time.perf_counter()
        retriever = load_retriever(store, state.device, state.rerank)
        log.info("indexes loaded: %d chunks in %.1f s", len(retriever.chunks), time.perf_counter() - t)
        t = time.perf_counter()
        warm_up(retriever, state.rerank)
        log.info("warm-up on %s in %.1f s; ready", state.device, time.perf_counter() - t)
        state.retriever = retriever
    except Exception as exc:
        log.exception("retriever failed to load")
        state.error = f"{type(exc).__name__}: {exc}"


def hit_out(hit) -> HitOut:
    d = hit.as_dict()
    d["url"] = d.pop("source")
    return HitOut(**d, act_title=hit.chunk.act_title)


def create_app(store: Path, device: str = "cpu", rerank: bool = False, retriever=None) -> FastAPI:
    state = State(device, rerank)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if retriever is None:
            threading.Thread(target=_start, args=(state, store), name="rag-load", daemon=True).start()
        else:
            warm_up(retriever, rerank)
            state.retriever = retriever
        yield

    app = FastAPI(title="npa-rag", summary="Hybrid search over Kazakhstan banking regulations", lifespan=lifespan)
    app.state.rag = state

    @app.get("/health")
    def health() -> Health:
        r = state.retriever
        return Health(ready=r is not None, chunks=len(r.chunks) if r is not None else 0, device=state.device,
                      rerank=state.rerank, error=state.error)

    @app.post("/search")
    def search(req: SearchRequest) -> SearchResponse:
        r = state.retriever
        if r is None:
            raise HTTPException(503, f"retriever not ready: {state.error}" if state.error else "retriever loading")
        if req.rerank and not state.rerank:
            raise HTTPException(400, "rerank requested but the server was started without --rerank")
        t0 = time.perf_counter()
        with state.lock:
            t1 = time.perf_counter()
            res = r.search(req.query, k=req.k, kinds=tuple(req.kinds), rerank=req.rerank)
        timings = {**res.timings_ms, "wait": (t1 - t0) * 1000, "total": (time.perf_counter() - t0) * 1000}
        return SearchResponse(query=res.query, hits=[hit_out(h) for h in res.hits],
                              timings_ms={k: round(v, 1) for k, v in timings.items()})

    return app
