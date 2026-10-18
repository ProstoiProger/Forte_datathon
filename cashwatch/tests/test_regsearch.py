import json
import logging
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cashwatch.analyzers.facts.schema import FLAG_KEYS, FactWindow, Norm
from cashwatch.regsearch import RegSearchClient, describe_facts_ru, truncate

LONG_TEXT = ("Банки обеспечивают хранение наличных денег в кладовых и (или) сейфовой комнате после их учета "
             "в автоматизированной цифровой системе в соответствии с внутренними документами банка. ") * 10

HITS = [
    {"citation": "Правила ведения кассовых операций, Глава 2, п. 22",
     "act_title": "Об утверждении Правил ведения кассовых операций",
     "act_requisites": "Постановление Правления НБРК от 29 ноября 2019 года № 231",
     "label": "п. 22", "part": "", "status": "Обновленный", "kind": "base", "rrf": 0.032, "ranks": {"bm25": 1},
     "rerank": None, "url": "https://adilet.zan.kz/rus/docs/V1900019686",
     "text": "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента."},
    {"citation": "Правила ведения кассовых операций, Глава 7, п. 74", "act_title": "",
     "act_requisites": "Постановление Правления НБРК от 29 ноября 2019 года № 231", "url": "", "text": LONG_TEXT},
    {"citation": "", "text": "hit without a citation is dropped"},
    "not a hit",
]


@dataclass
class Stub:
    status: int = 200
    body: bytes = b"{}"
    delay: float = 0.0
    requests: list[tuple[str, str, dict | None]] = field(default_factory=list)

    def reply(self, payload) -> None:
        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")


@pytest.fixture
def stub() -> Iterator[tuple[Stub, str]]:
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, body: dict | None) -> None:
            state.requests.append((self.command, self.path, body))
            time.sleep(state.delay)
            try:
                self.send_response(state.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(state.body)))
                self.end_headers()
                self.wfile.write(state.body)
            except OSError:
                pass  # the client gave up (timeout test)

        def do_GET(self) -> None:
            self._answer(None)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self._answer(json.loads(self.rfile.read(length).decode("utf-8")))

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def window(people: int = 1, **flags: bool) -> FactWindow:
    return FactWindow(index=0, start=0.0, end=8.0, frame_times=(0.0, 4.0), people=people,
                      flags={key: flags.get(key, False) for key in FLAG_KEYS})


def test_hits_mapped_to_norms(stub):
    state, url = stub
    state.reply({"query": "q", "hits": HITS, "timings_ms": {"total": 450.0}})
    norms = RegSearchClient(url).search("  Кассир покинул рабочее место  ")
    assert state.requests == [("POST", "/search", {"query": "Кассир покинул рабочее место", "k": 3})]
    assert len(norms) == 2
    first, second = norms
    assert first == Norm(
        title="Об утверждении Правил ведения кассовых операций",
        citation="Правила ведения кассовых операций, Глава 2, п. 22 "
                 "(Постановление Правления НБРК от 29 ноября 2019 года № 231)",
        quote=HITS[0]["text"], url="https://adilet.zan.kz/rus/docs/V1900019686", source="rag")
    assert second.title == "Постановление Правления НБРК от 29 ноября 2019 года № 231"   # no act title
    assert second.quote.endswith("…") and len(second.quote) <= 801
    assert LONG_TEXT.startswith(second.quote[:-1]) and LONG_TEXT[len(second.quote) - 1] == " "


def test_k_and_query_clamped(stub):
    state, url = stub
    state.reply({"hits": []})
    client = RegSearchClient(url)
    assert client.search("сейф " * 1000, k=50) == []
    assert client.search("   ") == []                      # blank query: no request
    (_, _, body), = state.requests
    assert body["k"] == 10 and len(body["query"]) <= 2000


def test_server_down_returns_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="cashwatch.regsearch"):
        assert RegSearchClient(f"http://127.0.0.1:{free_port()}", timeout=0.5).search("сейф открыт") == []
    assert "regulation search failed" in caplog.text


@pytest.mark.parametrize("status", [500, 503, 422])
def test_http_error_returns_empty(stub, caplog, status):
    state, url = stub
    state.status = status
    state.reply({"detail": "retriever loading"})
    with caplog.at_level(logging.WARNING, logger="cashwatch.regsearch"):
        assert RegSearchClient(url).search("сейф открыт") == []
    assert f"HTTP {status}" in caplog.text


@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe", b'{"hits": "nope"}', b"[1, 2]", b"null"])
def test_bad_body_returns_empty(stub, body):
    state, url = stub
    state.body = body
    assert RegSearchClient(url).search("сейф открыт") == []


def test_timeout_respected(stub):
    state, url = stub
    state.delay = 2.0
    state.reply({"hits": HITS})
    t = time.monotonic()
    assert RegSearchClient(url, timeout=0.3).search("сейф открыт") == []
    assert time.monotonic() - t < 1.5


def test_check(stub):
    state, url = stub
    client = RegSearchClient(url)
    state.reply({"ready": True, "chunks": 26182, "device": "cpu", "rerank": False, "error": None})
    ok, message = client.check()
    assert ok and "26182" in message and "cpu" in message
    assert state.requests[-1][:2] == ("GET", "/health")
    state.reply({"ready": False, "chunks": 0, "device": "cpu", "rerank": False, "error": None})
    assert client.check() == (False, f"regulation search at {url.rstrip('/')} not ready: loading")
    ok, message = RegSearchClient(f"http://127.0.0.1:{free_port()}", timeout=0.5).check()
    assert not ok and "unreachable" in message


def test_truncate_at_word_boundary():
    assert truncate("короткий текст", 800) == "короткий текст"
    assert truncate("слово " * 10, 20) == "слово слово слово…"
    assert truncate("x" * 30, 20) == "x" * 20 + "…"            # no whitespace: hard cut


def test_describe_leaving_with_cash_and_safe():
    text = describe_facts_ru(window(3, cashier_at_workplace=True, cashier_leaving=True, cash_on_desk=True,
                                    safe_open=True))
    assert text == ("В помещении кассового узла 3 человека. Кассовый работник покидает рабочее место; "
                    "наличные деньги оставлены на столе без присмотра; сейф открыт.")


def test_describe_cashier_at_work_with_visitor():
    text = describe_facts_ru(window(2, cashier_at_workplace=True, cash_on_desk=True, visitor_present=True))
    assert text == ("В помещении кассового узла 2 человека. Кассовый работник находится на рабочем месте; "
                    "наличные деньги лежат на столе; в помещении присутствуют посторонние лица.")


def test_describe_empty_room_and_plurals():
    assert describe_facts_ru(window(0, cash_on_desk=True)) == (
        "В помещении кассового узла нет людей. Кассовый работник отсутствует на рабочем месте; "
        "наличные деньги оставлены на столе без присмотра.")
    assert describe_facts_ru(window(1)) == "В помещении кассового узла 1 человек."
    assert describe_facts_ru(window(5)) == "В помещении кассового узла 5 человек."
    assert describe_facts_ru(window(12)) == "В помещении кассового узла 12 человек."
    assert describe_facts_ru(window(22)) == "В помещении кассового узла 22 человека."


def test_describe_items_and_unknown_flags():
    w = FactWindow(index=1, start=0.0, end=8.0, frame_times=(0.0,), people=1,
                   flags={"cashier_at_workplace": True, "phone_in_hand": True, "personal_items_on_desk": True,
                          "drinking": True, "door_open": True, "made_up_flag": True})
    assert describe_facts_ru(w) == (
        "В помещении кассового узла 1 человек. Кассовый работник находится на рабочем месте; дверь кассы открыта; "
        "кассовый работник пользуется мобильным телефоном (средством связи); "
        "кассовый работник употребляет напитки на рабочем месте; "
        "личные вещи (сумка, верхняя одежда) на рабочем месте кассового работника.")
