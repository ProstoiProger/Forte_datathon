import json
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cashwatch.analyzers.base import AnalyzerError, NullAnalyzer
from cashwatch.analyzers.fallback import FallbackAnalyzer
from cashwatch.analyzers.stub import MotionStubAnalyzer
from cashwatch.analyzers.vlm import (
    Frame,
    OllamaClient,
    VlmAnalyzer,
    VlmCallError,
    build_user_prompt,
    make_windows,
    merge_events,
    parse_response,
    response_schema,
)
from cashwatch.analyzers.base import Event
from cashwatch.contract import validate_result
from cashwatch.main import create_app
from cashwatch.rules import load_registry
from conftest import submit, wait_for, write_video

TYPE_IDS = sorted(t.type_id for t in load_registry().all() if t.type_id != "stub_motion")


class FakeClient:
    def __init__(self, reply) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, system, user, images, schema):
        call = {"system": system, "user": user, "images": images, "schema": schema, "n": len(self.calls)}
        self.calls.append(call)
        result = self.reply(call)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, str) else json.dumps(result)

    def check(self):
        return True, "fake"


def events_reply(*items):
    return lambda call: {"events": list(items)}


@pytest.fixture(scope="module")
def ten_second_video(tmp_path_factory) -> Path:
    return write_video(tmp_path_factory.mktemp("vlm") / "ten.mov", seconds=10.0)


def analyzer(reply, **kw) -> tuple[VlmAnalyzer, FakeClient]:
    client = FakeClient(reply)
    return VlmAnalyzer(client, max_side=160, **kw), client


def test_windows_cover_video_with_overlap() -> None:
    times = [i * 0.5 for i in range(20)]
    windows = make_windows(times, 10.0, 6.0, 3.0, 12)
    assert len(windows) == 3
    assert windows[0][0] == 0 and windows[-1][-1] == 19
    assert set().union(*map(set, windows)) == set(range(20))
    assert all(len(w) <= 12 for w in windows)
    assert make_windows([0.0, 0.5, 1.0], 1.5, 6.0, 3.0, 12) == [[0, 1, 2]]


def test_windows_subsample_to_max_frames() -> None:
    times = [i * 0.25 for i in range(24)]
    (window,) = make_windows(times, 6.0, 6.0, 3.0, 12)
    assert len(window) == 12 and window[0] == 0 and window[-1] == 23


def test_schema_enumerates_registry_types() -> None:
    schema = response_schema(TYPE_IDS, 7)
    item = schema["properties"]["events"]["items"]["properties"]
    assert item["type_id"]["enum"] == TYPE_IDS
    assert "stub_motion" not in item["type_id"]["enum"]
    assert item["frame"]["maximum"] == 7


def test_prompt_lists_types_and_frame_times() -> None:
    types = [t for t in load_registry().all() if t.verified_citations]
    prompt = build_user_prompt([Frame(1.0, "a"), Frame(1.5, "b")], types)
    assert "Frame 2: t=1.5s" in prompt
    for t in types:
        assert t.type_id in prompt and t.title in prompt


def test_model_frame_maps_to_timestamp(ten_second_video: Path) -> None:
    reply = lambda call: {"events": [{"type_id": "unauthorized_entry", "frame": 3, "confidence": 0.9, "observation": "Входит мужчина."}]} if call["n"] == 0 else {"events": []}
    vlm, client = analyzer(reply)
    events = vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    assert len(client.calls) == 3
    assert len(client.calls[0]["images"]) == 12
    (event,) = events
    assert event.type_id == "unauthorized_entry"
    assert event.timecode_sec == pytest.approx(1.0, abs=0.05)
    assert [r.timestamp_sec for r in event.evidence] == pytest.approx([0.5, 1.0, 1.5], abs=0.05)
    assert vlm.last_stats["windows"] == 3 and vlm.last_stats["failed"] == 0


def test_overlapping_windows_merge_same_type(ten_second_video: Path) -> None:
    def reply(call):
        frame = 9 if call["n"] == 0 else 3
        return {"events": [
            {"type_id": "safe_left_open", "frame": frame, "confidence": 0.6 + 0.1 * call["n"], "observation": "Сейф открыт."},
            {"type_id": "phone_use", "frame": frame, "confidence": 0.7, "observation": "Телефон."},
        ]}

    vlm, _ = analyzer(reply)
    events = vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    by_type = {e.type_id: e for e in events}
    assert sorted(by_type) == ["phone_use", "safe_left_open"]
    assert len(events) == 2
    assert by_type["safe_left_open"].confidence == pytest.approx(0.8)


def test_merge_keeps_distant_events_apart() -> None:
    ev = lambda t, c: Event("door_left_open", t, c, "")
    merged = merge_events([ev(1.0, 0.5), ev(3.0, 0.9), ev(20.0, 0.6)], gap=4.0)
    assert [(e.timecode_sec, e.confidence) for e in merged] == [(3.0, 0.9), (20.0, 0.6)]


@pytest.mark.parametrize("content", [
    "not json",
    "[]",
    json.dumps({"events": "none"}),
    json.dumps({"events": [{"type_id": "bogus", "frame": 1, "confidence": 0.9, "observation": ""}]}),
    json.dumps({"events": [{"type_id": "phone_use", "frame": 99, "confidence": 0.9, "observation": ""}]}),
    json.dumps({"events": [{"type_id": "phone_use", "frame": "x", "confidence": "high", "observation": ""}]}),
    json.dumps({"events": [None, 5, "a"]}),
])
def test_garbage_model_output_is_ignored(content: str) -> None:
    frames = [Frame(0.0, "a"), Frame(0.5, "b")]
    assert parse_response(content, frames, set(TYPE_IDS)) == []


def test_confidence_is_clamped() -> None:
    frames = [Frame(0.0, "a")]
    content = json.dumps({"events": [{"type_id": "phone_use", "frame": 1, "confidence": 7, "observation": " x "}]})
    (event,) = parse_response(content, frames, set(TYPE_IDS))
    assert event.confidence == 1.0 and event.observation == "x"


def test_failed_window_is_skipped_and_others_kept(ten_second_video: Path) -> None:
    def reply(call):
        if call["n"] in (0, 1):
            return VlmCallError("boom")
        return {"events": [{"type_id": "personal_items", "frame": 2, "confidence": 0.8, "observation": "Сумка на столе."}]}

    vlm, client = analyzer(reply)
    events = vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    assert len(client.calls) == 4
    assert [e.type_id for e in events] == ["personal_items"]
    assert vlm.last_stats["failed"] == 1


def test_retry_recovers_single_failure(ten_second_video: Path) -> None:
    reply = lambda call: VlmCallError("timeout") if call["n"] == 0 else {"events": []}
    vlm, client = analyzer(reply)
    vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    assert len(client.calls) == 4
    assert vlm.last_stats["failed"] == 0


def test_total_failure_raises_and_fallback_takes_over(ten_second_video: Path) -> None:
    vlm, _ = analyzer(lambda call: VlmCallError("ollama down"))
    with pytest.raises(AnalyzerError):
        vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    combined = FallbackAnalyzer(vlm, NullAnalyzer())
    assert combined.analyze(ten_second_video, lambda p: None, time.monotonic() + 60) == []
    assert combined.fallbacks == 1 and combined.name == "vlm" and combined.default_threshold == 0.5
    with_stub = FallbackAnalyzer(vlm, MotionStubAnalyzer())
    assert with_stub.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)


def test_deadline_stops_before_calling_model(ten_second_video: Path) -> None:
    vlm, client = analyzer(events_reply())
    assert vlm.analyze(ten_second_video, lambda p: None, time.monotonic()) == []
    assert client.calls == []


def test_progress_reaches_one(ten_second_video: Path) -> None:
    seen = []
    vlm, _ = analyzer(events_reply())
    vlm.analyze(ten_second_video, seen.append, time.monotonic() + 60)
    assert seen[-1] == pytest.approx(1.0)


def service(make_settings, confidence: float, **settings):
    reply = lambda call: {"events": [{"type_id": "unauthorized_entry", "frame": 5, "confidence": confidence, "observation": "В помещение входит мужчина в темной одежде."}]} if call["n"] == 0 else {"events": []}
    return create_app(make_settings(**settings), analyzer=VlmAnalyzer(FakeClient(reply), max_side=160))


def test_service_reports_violation_with_verified_rule(make_settings, ten_second_video: Path) -> None:
    with TestClient(service(make_settings, 0.9)) as client:
        body = wait_for(client, submit(client, ten_second_video).json()["job_id"])
        health = client.get("/api/v1/health").json()
    assert health["ready"] is True and health["analyzer"] == "vlm"
    assert validate_result(body, duration=10.0) == []
    (violation,) = body["violations"]
    assert violation["type"] == "Посторонний в кассовом помещении"
    assert violation["timecode_sec"] == pytest.approx(2.0, abs=0.1)
    assert "Основание: п. 75" in violation["description"]
    assert len(violation["evidence_frames"]) == 3


def test_service_threshold_drops_weak_detection(make_settings, ten_second_video: Path) -> None:
    with TestClient(service(make_settings, 0.3)) as client:
        body = wait_for(client, submit(client, ten_second_video).json()["job_id"])
    assert body["verdict"] == "no_violation"
    with TestClient(service(make_settings, 0.3, min_confidence=0.2)) as client:
        body = wait_for(client, submit(client, ten_second_video).json()["job_id"])
    assert body["verdict"] == "violation"


def test_service_survives_model_outage(make_settings, ten_second_video: Path) -> None:
    vlm = VlmAnalyzer(FakeClient(lambda call: VlmCallError("down")), max_side=160)
    with TestClient(create_app(make_settings(), analyzer=FallbackAnalyzer(vlm, NullAnalyzer()))) as client:
        body = wait_for(client, submit(client, ten_second_video).json()["job_id"])
    assert body["status"] == "done" and body["verdict"] == "no_violation"


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_client_sends_schema_and_images(monkeypatch) -> None:
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return _Response({"message": {"role": "assistant", "content": "{\"events\": []}"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OllamaClient("http://gpu:11434/", "qwen3-vl:8b")
    schema = response_schema(TYPE_IDS, 2)
    assert client.chat("sys", "user", ["AAA", "BBB"], schema) == "{\"events\": []}"
    body = sent["body"]
    assert sent["url"] == "http://gpu:11434/api/chat"
    assert body["model"] == "qwen3-vl:8b" and body["stream"] is False and body["format"] == schema
    assert body["options"]["temperature"] == 0
    assert body["messages"][1]["images"] == ["AAA", "BBB"]


def test_ollama_client_errors_become_vlm_errors(monkeypatch) -> None:
    def broken(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", broken)
    with pytest.raises(VlmCallError):
        OllamaClient("http://localhost:1", "m").chat("s", "u", [], {})
    ok, detail = OllamaClient("http://localhost:1", "m").check()
    assert not ok and "unreachable" in detail


def test_ollama_check_detects_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Response({"models": [{"name": "llama3:8b"}]}))
    ok, detail = OllamaClient("http://x", "qwen3-vl:8b").check()
    assert not ok and "not pulled" in detail
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Response({"models": [{"name": "qwen3-vl:8b"}]}))
    assert OllamaClient("http://x", "qwen3-vl:8b").check()[0]
