import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from cashwatch.analyzers.base import Event
from cashwatch.analyzers.vlm import (
    Reference,
    VlmAnalyzer,
    cluster_events,
    load_references,
    make_windows,
    plan_windows,
    score_events,
)
from cashwatch.transforms import TRANSFORMS, apply, transform_video
from conftest import write_video

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingClient:
    def __init__(self, reply=None) -> None:
        self.calls: list[dict] = []
        self.reply = reply or (lambda call: {"events": []})

    def chat(self, system, user, images, schema):
        call = {"user": user, "images": images, "schema": schema, "n": len(self.calls)}
        self.calls.append(call)
        return json.dumps(self.reply(call))


@pytest.fixture(scope="module")
def ten_second_video(tmp_path_factory) -> Path:
    return write_video(tmp_path_factory.mktemp("impr") / "ten.mov", seconds=10.0)


@pytest.mark.parametrize("duration,expected", [(30.0, (6.0, 3.0)), (70.0, (6.0, 64.0 / 19)), (145.0, (7.25, 7.25))])
def test_plan_windows_caps_long_videos(duration: float, expected: tuple[float, float]) -> None:
    window, stride = plan_windows(duration, 6.0, 3.0, 20)
    assert (window, stride) == pytest.approx(expected)
    assert stride <= window
    times = [i * 0.5 for i in range(int(duration * 2))]
    windows = make_windows(times, duration, window, stride, 12)
    assert len(windows) <= 20
    starts = [k * stride for k in range(len(windows))]
    assert all(any(s <= t < s + window for s in starts) for t in times)


def test_plan_windows_keeps_short_videos() -> None:
    assert plan_windows(4.0, 6.0, 3.0, 20) == (6.0, 3.0)
    assert plan_windows(300.0, 6.0, 3.0, 1) == (6.0, 3.0)


def _event(t: float, conf: float, window: int) -> Event:
    return Event("phone_use", t, conf, "obs", window=window)


def test_agreement_score_rewards_repeated_detection() -> None:
    spans = [(0.0, 5.5), (3.0, 8.5)]
    single = score_events(cluster_events([_event(4.0, 0.8, 0)], 4.0), spans)
    double = score_events(cluster_events([_event(4.0, 0.8, 0), _event(4.5, 0.6, 1)], 4.0), spans)
    raw = score_events(cluster_events([_event(4.0, 0.8, 0)], 4.0), spans, mode="max")
    assert single[0].confidence == pytest.approx(0.6)
    assert (single[0].votes, single[0].coverage, single[0].raw_confidence) == (1, 2, 0.8)
    assert double[0].confidence == pytest.approx(0.8) and double[0].votes == 2
    assert raw[0].confidence == pytest.approx(0.8)


def test_agreement_score_full_support_when_only_one_window_covers() -> None:
    scored = score_events(cluster_events([_event(1.0, 0.9, 0)], 4.0), [(0.0, 5.5), (3.0, 8.5)])
    assert scored[0].coverage == 1 and scored[0].confidence == pytest.approx(0.9)


def refs() -> list[Reference]:
    return [
        Reference("unauthorized_entry", "REF_A", "X-4", "s01"),
        Reference("normal", "REF_N", "X-3", "s01"),
        Reference("phone_use", "REF_P", "X-2", "s02"),
        Reference("not_a_type", "REF_X", "X-9", "s03"),
    ]


def test_references_are_sent_before_frames(ten_second_video: Path) -> None:
    reply = lambda call: {"events": [{"type_id": "phone_use", "frame": 3, "confidence": 0.9, "observation": "x"}]} if call["n"] == 0 else {"events": []}
    client = RecordingClient(reply)
    vlm = VlmAnalyzer(client, max_side=160, references=refs())
    events = vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    first = client.calls[0]
    assert first["images"][:3] == ["REF_A", "REF_N", "REF_P"]
    assert len(first["images"]) == 3 + 12
    assert "Reference image 1: unauthorized_entry" in first["user"]
    assert "Reference image 2: normal" in first["user"]
    assert "Frame 1 (image 4)" in first["user"]
    assert first["schema"]["properties"]["events"]["items"]["properties"]["frame"]["maximum"] == 12
    assert events[0].timecode_sec == pytest.approx(1.0, abs=0.05)
    assert vlm.last_stats["references"] == 3


def test_reference_groups_can_be_excluded_and_limited(ten_second_video: Path) -> None:
    client = RecordingClient()
    vlm = VlmAnalyzer(client, max_side=160, references=refs(), max_references=1)
    assert [r.image for r in vlm.active_references()] == ["REF_A"]
    vlm.exclude_reference_groups = {"s01"}
    assert [r.image for r in vlm.active_references()] == ["REF_P"]
    vlm.exclude_reference_groups = {"s01", "s02"}
    vlm.analyze(ten_second_video, lambda p: None, time.monotonic() + 60)
    assert client.calls and all(not any(i.startswith("REF_") for i in c["images"]) for c in client.calls)
    assert "Reference image" not in client.calls[0]["user"]


def _jpg(path: Path, color: int) -> None:
    cv2.imwrite(str(path), np.full((80, 60, 3), color, np.uint8))


def test_load_references_with_and_without_manifest(tmp_path: Path) -> None:
    _jpg(tmp_path / "phone_use__1.jpg", 50)
    _jpg(tmp_path / "normal__1.jpg", 200)
    loose = load_references(tmp_path)
    assert sorted(r.type_id for r in loose) == ["normal", "phone_use"]
    (tmp_path / "manifest.json").write_text(json.dumps([{"type_id": "phone_use", "file": "phone_use__1.jpg", "source_video": "X-2", "group": "s01"}]), encoding="utf-8")
    (ref,) = load_references(tmp_path)
    assert (ref.type_id, ref.source, ref.group) == ("phone_use", "X-2", "s01")
    assert ref.image


def test_build_references_skips_holdout_and_uncertain(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("a", "b", "c", "d"):
        write_video(videos / f"{name}.mov", seconds=4.0)
    labels = {"videos": [
        {"video": "A", "file": "a.mov", "group": "s01", "label": "violation", "events": [{"t": 2.0, "type": "phone_use", "type_status": "likely"}]},
        {"video": "B", "file": "b.mov", "group": "s11", "label": "violation", "events": [{"t": 2.0, "type": "food_drink", "type_status": "confirmed"}]},
        {"video": "C", "file": "c.mov", "group": "s02", "label": "violation", "events": [{"t": 1.0, "type": "door_left_open", "type_status": "uncertain"}]},
        {"video": "D", "file": "d.mov", "group": "s03", "label": "no_violation", "events": []},
    ]}
    build_references = load_script("build_references")
    manifest = build_references.build(labels, videos, tmp_path / "refs", {"s11"}, {"confirmed", "likely"}, 1, 1, 256)
    assert [(m["type_id"], m["source_video"]) for m in manifest] == [("phone_use", "A"), ("normal", "D")]
    loaded = load_references(tmp_path / "refs")
    assert [r.type_id for r in loaded] == ["phone_use", "normal"]
    image = cv2.imread(str(tmp_path / "refs" / "phone_use__1.jpg"))
    assert max(image.shape[:2]) <= 256


def test_every_transform_keeps_frame_shape() -> None:
    frame = np.zeros((120, 80, 3), np.uint8)
    frame[:, :20] = 255
    for name in TRANSFORMS:
        out = apply(frame, name)
        assert out.shape == frame.shape and out.dtype == frame.dtype
    flipped = apply(frame, "flip")
    assert flipped[:, -20:].mean() == 255 and flipped[:, :20].mean() == 0
    with pytest.raises(ValueError):
        apply(frame, "sepia")


def test_transform_video_writes_readable_copy_and_caches(tmp_path: Path) -> None:
    src = write_video(tmp_path / "src.mov", seconds=2.0)
    dst = transform_video(src, tmp_path / "out" / "warp.mov", "warp")
    cap = cv2.VideoCapture(str(dst))
    n_dst = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap = cv2.VideoCapture(str(src))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert abs(n_dst - n_src) <= 1
    stamp = dst.stat().st_mtime
    assert transform_video(src, dst, "warp") == dst and dst.stat().st_mtime == stamp


def _labels(tmp_path: Path) -> Path:
    videos = tmp_path / "videos"
    videos.mkdir()
    write_video(videos / "a.mov", seconds=4.0)
    write_video(videos / "b.mov", seconds=4.0)
    labels = {"videos": [
        {"video": "A", "file": "a.mov", "group": "s01", "label": "violation", "events": [{"t": 2.5, "type": None}]},
        {"video": "B", "file": "b.mov", "group": "s02", "label": "no_violation", "events": []},
    ]}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    return path


def test_evaluate_runs_on_transformed_videos(tmp_path: Path) -> None:
    labels = _labels(tmp_path)
    evaluate = load_script("evaluate")
    args = ["run", "--name", "flip", "--labels", str(labels), "--videos", str(tmp_path / "videos"), "--runs-dir", str(tmp_path / "runs"),
            "--analyzer", "stub", "--transform", "flip", "--transform-cache", str(tmp_path / "cache")]
    assert evaluate.main(args) == 0
    assert (tmp_path / "cache" / "flip" / "A.mov").exists()
    config = json.loads((tmp_path / "runs" / "flip" / "config.json").read_text(encoding="utf-8"))
    assert config["transform"] == "flip"
    record = json.loads((tmp_path / "runs" / "flip" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"votes", "coverage", "raw_conf"} <= record["events"][0].keys()


def test_queue_runs_experiments_and_builds_leaderboard(tmp_path: Path) -> None:
    labels = _labels(tmp_path)
    spec = {
        "defaults": ["--labels", str(labels), "--videos", str(tmp_path / "videos"), "--analyzer", "stub"],
        "experiments": [
            {"name": "peak", "args": []},
            {"name": "silent", "args": ["--stub-mode", "none"]},
        ],
    }
    spec_path = tmp_path / "experiments.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    run_queue = load_script("run_queue")
    runs = tmp_path / "runs"
    assert run_queue.main([str(spec_path), "--runs-dir", str(runs)]) == 0
    rows = run_queue.leaderboard(runs)
    assert sorted(r["run"] for r in rows) == ["peak", "silent"]
    board = (runs / "leaderboard.md").read_text(encoding="utf-8")
    assert "| peak |" in board and "| silent |" in board
    assert run_queue.main([str(spec_path), "--runs-dir", str(runs)]) == 0
    assert "skip peak: already scored" in (runs / "queue.log").read_text(encoding="utf-8")
    assert run_queue.main(["--leaderboard-only", "--runs-dir", str(runs)]) == 0
