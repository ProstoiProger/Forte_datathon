import http.client
import io
import json
import types
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from cashwatch.analyzers.facts import perception
from cashwatch.analyzers.facts.perception import (
    FACTS_SYSTEM,
    Perceiver,
    facts_schema,
    scrub_legal,
    verify_schema,
    window_motion,
)
from cashwatch.analyzers.facts.schema import FLAG_DESCRIPTIONS, FLAG_KEYS, PEOPLE_DESCRIPTION, CaseRef, Norm
from cashwatch.analyzers.vlm import Frame, OllamaClient, VlmCallError
from cashwatch.video import encode_jpeg_b64


class FakeClient:
    def __init__(self, *replies, clock=None, seconds_per_call: float = 0.0) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.clock = clock
        self.seconds_per_call = seconds_per_call

    def chat(self, system, user, images, schema, **kwargs):
        self.calls.append({"system": system, "user": user, "images": images, "schema": schema, "kwargs": kwargs})
        if self.clock:
            self.clock.now += self.seconds_per_call
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def image(shade: int = 200, box: tuple[int, int] | None = None) -> str:
    frame = np.full((288, 160, 3), shade, np.uint8)
    if box:
        x, y = box
        cv2.rectangle(frame, (x, y), (x + 60, y + 80), (20, 20, 20), -1)
    return encode_jpeg_b64(frame, quality=90, max_side=288)


def frames(n: int = 4, start: float = 12.0) -> list[Frame]:
    return [Frame(start + i, f"F{i + 1}") for i in range(n)]


def all_flags(**overrides) -> dict:
    return {**{k: False for k in FLAG_KEYS}, "people": 1, "change_frame": 0, **overrides}


NORM = Norm("Сейф оставлен открытым", "п. 22, Правила (постановление НБРК № 231)", "Хранилище закрывается кассиром при выходе.")
NORMS = [
    Norm("Правила ведения кассовых операций", "п. 5, Правила № 1", "Кассир обеспечивает сохранность наличных денег."),
    Norm("Инструкция по охране", "п. 9, Инструкция № 2", "Двери кассы держатся закрытыми."),
]
REFS = [
    CaseRef("violation", "safe_left_open", "R1", "Кассир ушёл, сейф открыт.", "X-24#e1", "g1"),
    CaseRef("normal", None, "R2", "Кассир закрывает сейф и уходит.", "X-25@012.00", "g2"),
]


# ---------------------------------------------------------------- facts

def test_facts_schema_has_every_flag_and_counts() -> None:
    schema = facts_schema(4)
    props = schema["properties"]
    assert set(props) == {*FLAG_KEYS, "people", "change_frame"}
    assert set(schema["required"]) == set(props)
    assert all(props[k]["type"] == "boolean" for k in FLAG_KEYS)
    assert props["people"] == {"type": "integer", "minimum": 0}
    assert props["change_frame"] == {"type": "integer", "minimum": 0, "maximum": 4}


def test_facts_prompt_lists_frames_and_verbatim_descriptions() -> None:
    client = FakeClient(all_flags())
    window = Perceiver(client).facts(frames(), 3, 12.0, 16.0)
    assert window is not None
    (call,) = client.calls
    assert call["system"] == FACTS_SYSTEM
    assert call["images"] == ["F1", "F2", "F3", "F4"]
    assert "Frame 1: t=12.0s" in call["user"] and "Frame 4: t=15.0s" in call["user"]
    assert "ANY" in call["user"] and "JSON" in call["user"]
    for key, text in FLAG_DESCRIPTIONS.items():
        assert f"- {key}: {text}" in call["user"]
    assert PEOPLE_DESCRIPTION in call["user"]
    assert call["schema"]["properties"]["change_frame"]["maximum"] == 4


def test_facts_fill_window() -> None:
    reply = all_flags(safe_open=True, phone_in_hand=True, people=2, change_frame=3)
    window = Perceiver(FakeClient(reply)).facts(frames(), 7, 12.0, 16.0)
    assert window.index == 7 and (window.start, window.end) == (12.0, 16.0)
    assert window.frame_times == (12.0, 13.0, 14.0, 15.0)
    assert window.flags == {k: k in {"safe_open", "phone_in_hand"} for k in FLAG_KEYS}
    assert window.people == 2 and window.change_frame == 3
    assert window.motion == 0.0  # fake images are not JPEGs


def test_facts_missing_and_invalid_fields_are_defaulted_and_clamped() -> None:
    reply = {"safe_open": "true", "cash_on_desk": 1, "door_open": None, "people": -3, "change_frame": 9}
    window = Perceiver(FakeClient(reply)).facts(frames(), 0, 0.0, 4.0)
    assert window.flags["safe_open"] is True
    assert window.flags["cash_on_desk"] is False and window.flags["door_open"] is False
    assert all(window.flags[k] is False for k in FLAG_KEYS if k != "safe_open")
    assert set(window.flags) == set(FLAG_KEYS)
    assert window.people == 0 and window.change_frame == 4
    window = Perceiver(FakeClient({"people": "many", "change_frame": "x"})).facts(frames(), 0, 0.0, 4.0)
    assert window.people == 0 and window.change_frame == 0
    window = Perceiver(FakeClient({"people": 2.0, "change_frame": -1})).facts(frames(), 0, 0.0, 4.0)
    assert window.people == 2 and window.change_frame == 0


@pytest.mark.parametrize("content", ["not json", "[]", "null", json.dumps({"events": []})])
def test_facts_unusable_answer_returns_none_after_retries(content: str) -> None:
    client = FakeClient(content)
    assert Perceiver(client, retries=2).facts(frames(), 0, 0.0, 4.0) is None
    assert len(client.calls) == 3


def test_facts_retry_after_call_error_and_bad_json() -> None:
    client = FakeClient(VlmCallError("timeout"), "garbage", all_flags(visitor_present=True, people=2))
    perceiver = Perceiver(client, retries=2)
    window = perceiver.facts(frames(), 0, 0.0, 4.0)
    assert window.flags["visitor_present"] and len(client.calls) == 3
    assert perceiver.stats["calls"] == 1 and perceiver.stats["failed"] == 0


def test_facts_all_attempts_fail() -> None:
    client = FakeClient(VlmCallError("down"))
    perceiver = Perceiver(client, retries=1)
    assert perceiver.facts(frames(), 0, 0.0, 4.0) is None
    assert len(client.calls) == 2 and perceiver.stats["failed"] == 1


def test_facts_without_frames_skips_call() -> None:
    client = FakeClient(all_flags())
    assert Perceiver(client).facts([], 0, 0.0, 4.0) is None and client.calls == []


def test_facts_motion_from_real_images() -> None:
    still = [Frame(i, image()) for i in range(4)]
    moving = [Frame(0, image()), Frame(1, image(box=(10, 40))), Frame(2, image(box=(80, 160)))]
    assert Perceiver(FakeClient(all_flags())).facts(still, 0, 0, 4).motion == 0.0
    assert Perceiver(FakeClient(all_flags())).facts(moving, 0, 0, 3).motion > 0.05


# ---------------------------------------------------------------- motion

def test_window_motion_identical_frames_is_zero() -> None:
    assert window_motion([image(), image(), image()]) == 0.0
    assert window_motion([image()]) == 0.0
    assert window_motion([]) == 0.0


def test_window_motion_changed_frames() -> None:
    one_box = window_motion([image(), image(box=(50, 100))])
    assert 0.05 < one_box < 0.3
    assert window_motion([image(), image(), image(box=(50, 100))]) == pytest.approx(one_box)
    assert window_motion([image(200), image(40)]) == pytest.approx(1.0)
    assert window_motion([image(200), image(210)]) == 0.0  # below the pixel threshold


def test_window_motion_ignores_undecodable_images() -> None:
    assert window_motion(["not-base64!!", image(), "AAAA"]) == 0.0
    assert window_motion([image(), "AAAA", image(box=(50, 100))]) > 0.05


# ---------------------------------------------------------------- verify

def verdict(**overrides) -> dict:
    return {"description_ru": "Кассир вышел, дверца сейфа открыта.", "confirmed": True, "frame": 2,
            "norm_index": 1, "title_ru": "", "confidence": 0.8, **overrides}


def test_verify_schema() -> None:
    schema = verify_schema(4, 2)
    props = schema["properties"]
    assert set(props) == {"confirmed", "frame", "description_ru", "confidence", "title_ru", "norm_index"}
    assert set(schema["required"]) == set(props)
    assert props["frame"]["maximum"] == 4 and props["norm_index"]["maximum"] == 2
    assert props["confidence"] == {"type": "number", "minimum": 0, "maximum": 1}


def test_verify_registry_norm() -> None:
    client = FakeClient(verdict(title_ru="лишнее"))
    result = Perceiver(client).verify(frames(), [NORM], [], "Сейф оставлен открытым")
    assert result.confirmed and result.frame == 2 and result.confidence == 0.8
    assert result.description == "Кассир вышел, дверца сейфа открыта."
    assert result.title is None and result.norm_index == 1
    (call,) = client.calls
    user = call["user"]
    assert 'show the violation "Сейф оставлен открытым"' in user
    assert NORM.citation in user and f'"{NORM.quote}"' in user and "Norm 1:" in user
    assert "Frame 1: t=12.0s" in user and "Frame 4: t=15.0s" in user
    assert "reference" not in user.lower()
    assert "Normal work is not a violation" in user and "locking it" in user and "nothing left on the desk" in user
    assert call["images"] == ["F1", "F2", "F3", "F4"]
    assert call["schema"]["properties"]["norm_index"]["maximum"] == 1


def test_verify_search_norms_pick_norm_and_title() -> None:
    client = FakeClient(verdict(norm_index=2, title_ru="Дверь кассы оставлена открытой", frame=3))
    result = Perceiver(client).verify(frames(), NORMS, [], None)
    assert result.confirmed and result.norm_index == 2 and result.frame == 3
    assert result.title == "Дверь кассы оставлена открытой"
    user = client.calls[0]["user"]
    assert "will not apply" in user and "which one" in user
    assert "Norm 1: Правила ведения кассовых операций" in user and "Norm 2: Инструкция по охране" in user
    assert "(1-2), 0 if none" in user


def test_verify_not_confirmed_drops_frame_and_norm() -> None:
    reply = verdict(confirmed=False, frame=3, norm_index=2, title_ru="", confidence=0.1, description_ru="Кассир считает деньги.")
    result = Perceiver(FakeClient(reply)).verify(frames(), NORMS, [], None)
    assert not result.confirmed and result.frame == 0 and result.norm_index is None and result.title is None
    assert result.confidence == 0.1 and result.description == "Кассир считает деньги."


@pytest.mark.parametrize("frame", [0, 9, "x"])
def test_verify_confirmed_without_valid_frame_uses_middle(frame) -> None:
    result = Perceiver(FakeClient(verdict(frame=frame))).verify(frames(5), [NORM], [], "Сейф оставлен открытым")
    assert result.frame == 3


def test_verify_clamps_and_ignores_bad_norm_index() -> None:
    result = Perceiver(FakeClient(verdict(norm_index=7, confidence=3))).verify(frames(), NORMS, [], None)
    assert result.norm_index is None and result.confidence == 1.0
    result = Perceiver(FakeClient(verdict(norm_index=0, confidence="high"))).verify(frames(), NORMS, [], None)
    assert result.norm_index is None and result.confidence == 0.0


def test_references_come_first_and_numbering_matches() -> None:
    client = FakeClient(verdict())
    Perceiver(client).verify(frames(3), [NORM], REFS, "Сейф оставлен открытым")
    (call,) = client.calls
    assert call["images"] == ["R1", "R2", "F1", "F2", "F3"]
    user = call["user"]
    assert "Images 1-2 are references from OTHER recordings" in user and "do not judge them" in user
    assert "Image 1 - reference: example of a confirmed violation. Кассир ушёл, сейф открыт." in user
    assert "Image 2 - reference: example of normal work. Кассир закрывает сейф и уходит." in user
    assert "Images 3-5 are the candidate frames" in user
    assert "Frame 1 (image 3): t=12.0s" in user and "Frame 3 (image 5): t=14.0s" in user
    assert user.index("Image 2 - reference") < user.index("Frame 1 (image 3)")


@pytest.mark.parametrize("content", ["not json", "[]", json.dumps({"confirmed": "maybe"}), json.dumps({"frame": 1})])
def test_verify_unusable_answer_returns_none_after_retries(content: str) -> None:
    client = FakeClient(content)
    assert Perceiver(client, retries=1).verify(frames(), [NORM], REFS, "Сейф оставлен открытым") is None
    assert len(client.calls) == 2


def test_verify_retries_call_errors() -> None:
    client = FakeClient(VlmCallError("timeout"), verdict())
    assert Perceiver(client, retries=1).verify(frames(), [NORM], [], "Сейф").confirmed
    client = FakeClient(VlmCallError("timeout"))
    assert Perceiver(client, retries=0).verify(frames(), [NORM], [], "Сейф") is None
    assert len(client.calls) == 1


def test_verify_without_frames_or_question_skips_call() -> None:
    client = FakeClient(verdict())
    assert Perceiver(client).verify([], [NORM], [], "Сейф") is None
    assert Perceiver(client).verify(frames(), [], [], None) is None
    assert client.calls == []


def test_verify_long_quote_is_shortened() -> None:
    client = FakeClient(verdict())
    long_norm = Norm("Т", "п. 1", "слово " * 1000)
    Perceiver(client).verify(frames(), [long_norm], [], "Т")
    assert len(client.calls[0]["user"]) < 4000


def test_verify_prompt_guidance_on_clip_end_and_citations() -> None:
    client = FakeClient(verdict())
    Perceiver(client).verify(frames(), [NORM], [], "Сейф оставлен открытым")
    user = client.calls[0]["user"]
    assert "Frame times are seconds from the start of the clip; the last frame may be the end of the clip" in user
    assert "whether a state persists" in user
    assert "Do not cite or mention norms, clauses or documents in description_ru or title_ru." in user
    assert "number of the first candidate frame (1-4) in which the violation is visible, i.e. where it starts" in user


# ---------------------------------------------------------------- legal references in public text

@pytest.mark.parametrize("text, expected", [
    ("Кассир вышел, оставив сейф открытым. Это нарушение п. 22 Правил ведения кассовых операций.",
     "Кассир вышел, оставив сейф открытым."),
    ("Кассир вышел в нарушение п.22. Сейф остался открытым!", "Сейф остался открытым!"),
    ("Согласно постановлению НБРК № 231 кассир не должен уходить. Кассир ушёл.", "Кассир ушёл."),
    ("Нарушен пункт 5. Нарушен подпункт 3) Инструкции. Дверь открыта.", "Дверь открыта."),
    ("Это противоречит ст. 5 Закона о банках. Кассир держит телефон.", "Кассир держит телефон."),
    ("См. статью 12 Кодекса. Глава 3 приказа. Кассир пьёт кофе.", "Кассир пьёт кофе."),
    ("Требование параграфа 2 не выполнено. Нарушены правила.", ""),
    ("Кассир направился к выходу и не закрыл сейф. Главный кассир правильно считает деньги.",
     "Кассир направился к выходу и не закрыл сейф. Главный кассир правильно считает деньги."),
    ("", ""),
])
def test_scrub_legal_drops_sentences_with_legal_references(text: str, expected: str) -> None:
    assert scrub_legal(text) == expected


def test_verification_text_is_scrubbed() -> None:
    reply = verdict(description_ru="Кассир ушёл, сейф открыт. Это нарушает п. 22 Правил (постановление НБРК № 231).",
                    title_ru="Нарушение пункта 22 Правил")
    result = Perceiver(FakeClient(reply)).verify(frames(), NORMS, [], None)
    assert result.description == "Кассир ушёл, сейф открыт." and result.title is None
    reply = verdict(description_ru="В нарушение п. 22 Правил сейф открыт.", title_ru="Сейф оставлен открытым")
    result = Perceiver(FakeClient(reply)).verify(frames(), NORMS, [], None)
    assert result.description == "" and result.title == "Сейф оставлен открытым"


# ---------------------------------------------------------------- deadline

class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(perception, "time", types.SimpleNamespace(monotonic=fake.monotonic))
    return fake


def test_no_deadline_keeps_client_signature() -> None:
    client = FakeClient(all_flags())
    Perceiver(client).facts(frames(), 0, 0.0, 4.0)
    assert client.calls[0]["kwargs"] == {}


def test_deadline_caps_timeout(clock: Clock) -> None:
    client = FakeClient(all_flags(), verdict())
    client.timeout = 90.0
    Perceiver(client).facts(frames(), 0, 0.0, 4.0, deadline=clock.now + 300)
    assert client.calls[0]["kwargs"] == {"timeout": 90.0}
    Perceiver(client).verify(frames(), [NORM], [], "Сейф", deadline=clock.now + 30)
    assert client.calls[1]["kwargs"] == {"timeout": pytest.approx(28.0)}


def test_deadline_without_client_timeout_uses_time_left(clock: Clock) -> None:
    client = FakeClient(all_flags())
    Perceiver(client).facts(frames(), 0, 0.0, 4.0, deadline=clock.now + 10)
    assert client.calls[0]["kwargs"] == {"timeout": pytest.approx(8.0)}


def test_too_close_to_deadline_skips_call(clock: Clock) -> None:
    client = FakeClient(all_flags(), verdict())
    perceiver = Perceiver(client)
    assert perceiver.facts(frames(), 0, 0.0, 4.0, deadline=clock.now + 4.9) is None
    assert perceiver.verify(frames(), [NORM], [], "Сейф", deadline=clock.now - 1) is None
    assert client.calls == []
    assert perceiver.stats["skipped"] == 2 and perceiver.stats["failed"] == 0


def test_retry_stops_at_deadline(clock: Clock) -> None:
    client = FakeClient(VlmCallError("timed out"), clock=clock, seconds_per_call=20.0)
    client.timeout = 90.0
    perceiver = Perceiver(client, retries=3)
    assert perceiver.facts(frames(), 0, 0.0, 4.0, deadline=clock.now + 45) is None
    assert [c["kwargs"]["timeout"] for c in client.calls] == pytest.approx([43.0, 23.0, 3.0])
    assert perceiver.stats["failed"] == 1


# ---------------------------------------------------------------- OllamaClient think flag

class _Response:
    def __init__(self, payload: dict | bytes | Exception) -> None:
        self.payload = payload if isinstance(payload, (bytes, Exception)) else json.dumps(payload).encode()

    def read(self) -> bytes:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/api/chat", code, "error", {}, io.BytesIO(body.encode()))


def fake_server(monkeypatch, *outcomes) -> list[dict]:
    """Each outcome: an exception raised by urlopen, a _Response, or a JSON payload."""
    bodies: list[dict] = []

    def urlopen(request, timeout):
        bodies.append({**json.loads(request.data), "_timeout": timeout})
        outcome = outcomes[min(len(bodies), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, _Response) else _Response(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return bodies


OK = {"message": {"role": "assistant", "content": "{}"}}


def test_ollama_client_disables_thinking(monkeypatch) -> None:
    bodies = fake_server(monkeypatch, OK)
    assert OllamaClient("http://x", "qwen3.8:27b").chat("s", "u", [], {}) == "{}"
    assert bodies[0]["think"] is False


def test_ollama_client_timeout_per_call(monkeypatch) -> None:
    bodies = fake_server(monkeypatch, OK)
    client = OllamaClient("http://x", "m", timeout=90.0)
    client.chat("s", "u", [], {})
    client.chat("s", "u", [], {}, timeout=12.5)
    assert [b["_timeout"] for b in bodies] == [90.0, 12.5]


@pytest.mark.parametrize("body", ['{"error":"model does not support thinking"}', '{"error":"Invalid field THINK"}'])
def test_ollama_client_retries_without_think_on_400(monkeypatch, body: str) -> None:
    bodies = fake_server(monkeypatch, _http_error(400, body), OK, OK)
    client = OllamaClient("http://x", "llava:7b")
    assert client.chat("s", "u", ["A"], {"type": "object"}, timeout=30.0) == "{}"
    assert len(bodies) == 2
    assert bodies[0]["think"] is False and "think" not in bodies[1]
    assert {k: v for k, v in bodies[0].items() if k != "think"} == bodies[1]
    assert bodies[1]["_timeout"] == 30.0
    assert client.chat("s", "u", [], {}) == "{}"
    assert len(bodies) == 3 and "think" not in bodies[2] and not client.send_think


def test_ollama_client_other_400_is_not_retried(monkeypatch) -> None:
    bodies = fake_server(monkeypatch, _http_error(400, '{"error":"invalid image input"}'), OK)
    client = OllamaClient("http://x", "m")
    with pytest.raises(VlmCallError) as info:
        client.chat("s", "u", [], {})
    assert info.value.status == 400 and "invalid image" in info.value.body
    assert len(bodies) == 1 and client.send_think


def test_ollama_client_other_http_errors_are_not_retried(monkeypatch) -> None:
    bodies = fake_server(monkeypatch, _http_error(500, "boom"), OK)
    client = OllamaClient("http://x", "m")
    with pytest.raises(VlmCallError) as info:
        client.chat("s", "u", [], {})
    assert info.value.status == 500 and "HTTP 500" in str(info.value) and "boom" in str(info.value)
    assert len(bodies) == 1 and client.send_think


def test_ollama_client_keeps_think_when_retry_also_fails(monkeypatch) -> None:
    bodies = fake_server(monkeypatch, _http_error(400, "does not support thinking"), _http_error(400, "bad image"))
    client = OllamaClient("http://x", "m")
    with pytest.raises(VlmCallError) as info:
        client.chat("s", "u", [], {})
    assert info.value.status == 400 and len(bodies) == 2 and client.send_think


@pytest.mark.parametrize("response", [
    _Response(http.client.IncompleteRead(b"{\"mess")),
    _Response(b"\xff\xfe\xfa not utf-8"),
    _Response(b"not json"),
    _Response(b"[1, 2]"),
    _Response({"message": "text"}),
    http.client.RemoteDisconnected("closed"),
])
def test_ollama_client_transport_errors_become_vlm_errors(monkeypatch, response) -> None:
    fake_server(monkeypatch, response)
    with pytest.raises(VlmCallError):
        OllamaClient("http://x", "m").chat("s", "u", [], {})


def test_vlm_call_error_status_is_optional() -> None:
    error = VlmCallError("x")
    assert error.status is None and error.body == ""
