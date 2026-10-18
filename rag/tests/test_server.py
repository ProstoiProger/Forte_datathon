import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from npa_rag import server
from npa_rag.corpus import Chunk
from npa_rag.hybrid import Hit, Result
from npa_rag.server import WARMUP_QUERY, create_app


def chunk(n: int, kind: str = "base") -> Chunk:
    return Chunk(id=f"V{n}:00001", act_id=f"V{n}", kind=kind, status="Утративший силу" if kind == "repealed" else "Обновленный",
                 section="02_cash", act_title=f"Об утверждении Правил {n}",
                 act_requisites=f"Постановление Правления НБРК от 1 января 2020 года № {n}",
                 segment=f"Правила {n}", path=["Глава 2. Хранение"], unit="point", label="п. 22", part="",
                 text="Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.",
                 source=f"https://adilet.zan.kz/rus/docs/V{n}")


class FakeRetriever:
    def __init__(self) -> None:
        self.chunks = [chunk(1), chunk(2, "repealed"), chunk(3), chunk(4), chunk(5)]
        self.calls: list[dict] = []

    def search(self, query: str, k: int = 10, kinds=("base",), rerank: bool = False, **kw) -> Result:
        self.calls.append({"query": query, "k": k, "kinds": tuple(kinds), "rerank": rerank, **kw})
        pool = [c for c in self.chunks if c.kind in kinds]
        hits = [Hit(c, 1 / (61 + i), {"bm25": i + 1, "dense": i + 2}, 0.9 - i / 10 if rerank else None)
                for i, c in enumerate(pool[:k])]
        return Result(query, hits, {"bm25": 1.23, "embed_query": 440.0, "dense": 3.0})


@pytest.fixture
def fake() -> FakeRetriever:
    return FakeRetriever()


def client(fake: FakeRetriever, rerank: bool = False) -> TestClient:
    return TestClient(create_app(Path("unused"), device="cpu", rerank=rerank, retriever=fake))


def test_health_and_warm_up(fake):
    with client(fake) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ready": True, "chunks": 5, "device": "cpu", "rerank": False, "error": None}
    assert fake.calls[0]["query"] == WARMUP_QUERY


def test_search_happy_path(fake):
    with client(fake) as c:
        r = c.post("/search", json={"query": "  кассир покинул рабочее место  "})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "кассир покинул рабочее место"
    assert len(body["hits"]) == 3                                   # default k
    hit = body["hits"][0]
    assert hit["citation"] == "Правила 1, Глава 2. Хранение, п. 22"
    assert hit["act_title"] == "Об утверждении Правил 1"
    assert hit["act_requisites"].endswith("№ 1")
    assert hit["url"] == "https://adilet.zan.kz/rus/docs/V1"
    assert hit["text"].startswith("Кассовому работнику")
    assert hit["kind"] == "base" and hit["status"] == "Обновленный" and hit["label"] == "п. 22"
    assert hit["ranks"] == {"bm25": 1, "dense": 2} and hit["rerank"] is None and hit["rrf"] > 0
    assert {"bm25", "embed_query", "dense", "wait", "total"} <= set(body["timings_ms"])
    assert fake.calls[-1] == {"query": "кассир покинул рабочее место", "k": 3, "kinds": ("base",), "rerank": False}


def test_kinds_passed_through(fake):
    with client(fake) as c:
        r = c.post("/search", json={"query": "сейф", "k": 10, "kinds": ["base", "repealed"]})
    assert r.status_code == 200
    assert fake.calls[-1]["kinds"] == ("base", "repealed")
    assert [h["kind"] for h in r.json()["hits"]].count("repealed") == 1


@pytest.mark.parametrize("payload", [
    {"query": ""},
    {"query": "   "},
    {"query": "x" * 2001},
    {"query": "сейф", "k": 0},
    {"query": "сейф", "k": 11},
    {"query": "сейф", "kinds": []},
    {"query": "сейф", "kinds": ["amendment"]},
    {"query": "сейф", "top_k": 3},
    {},
])
def test_validation_errors(fake, payload):
    with client(fake) as c:
        r = c.post("/search", json=payload)
    assert r.status_code == 422, payload
    assert len(fake.calls) == 1                                     # only the warm-up reached the retriever


def test_rerank_refused_when_not_enabled(fake):
    with client(fake) as c:
        r = c.post("/search", json={"query": "сейф", "rerank": True})
    assert r.status_code == 400
    assert "--rerank" in r.json()["detail"]


def test_rerank_when_enabled(fake):
    with client(fake, rerank=True) as c:
        assert c.get("/health").json()["rerank"] is True
        r = c.post("/search", json={"query": "сейф", "k": 2, "rerank": True})
    assert r.status_code == 200
    assert [h["rerank"] for h in r.json()["hits"]] == [0.9, 0.8]
    assert fake.calls[0]["rerank"] is True and fake.calls[0]["depth"] == 2   # warm-up loads the reranker


def test_not_ready_when_loading_fails(monkeypatch):
    def broken(store, device, rerank):
        raise FileNotFoundError("store/chunks.jsonl")

    monkeypatch.setattr(server, "load_retriever", broken)
    with TestClient(create_app(Path("missing"))) as c:
        deadline = time.monotonic() + 5
        while c.get("/health").json()["error"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        health = c.get("/health").json()
        r = c.post("/search", json={"query": "сейф"})
    assert health["ready"] is False and health["chunks"] == 0
    assert "FileNotFoundError" in health["error"]
    assert r.status_code == 503
