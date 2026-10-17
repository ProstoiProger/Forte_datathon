import base64
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cashwatch.analyzers.base import Event, EvidenceRef
from cashwatch.contract import validate_result
from cashwatch.main import create_app
from conftest import HEADERS, submit, wait_for, write_video


def test_health_is_open(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_analyze_rejects_bad_key(client: TestClient, video_path: Path, headers: dict) -> None:
    assert submit(client, video_path, headers=headers).status_code == 401


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_jobs_rejects_bad_key(client: TestClient, video_path: Path, headers: dict) -> None:
    job_id = submit(client, video_path).json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).status_code == 401


def test_analyze_rejects_wrong_extension(client: TestClient, video_path: Path) -> None:
    assert submit(client, video_path, name="clip.avi").status_code == 415


def test_analyze_accepts_uppercase_extension(client: TestClient, video_path: Path) -> None:
    assert submit(client, video_path, name="IMG_0042.MOV").status_code == 200


def test_analyze_requires_file_field(client: TestClient) -> None:
    response = client.post("/api/v1/analyze", headers=HEADERS, files={"video": ("a.mov", b"x", "video/quicktime")})
    assert response.status_code == 422


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/jobs/doesnotexist", headers=HEADERS).status_code == 404


def test_violation_result_matches_contract(client: TestClient, video_path: Path) -> None:
    response = submit(client, video_path)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert isinstance(job_id, str) and job_id
    body = wait_for(client, job_id)
    assert validate_result(body, duration=4.0) == []
    assert body["job_id"] == job_id
    assert body["verdict"] == "violation"
    event = body["violations"][0]
    assert 1.5 <= event["timecode_sec"] <= 3.5
    jpeg = base64.b64decode(event["evidence_frames"][0]["image_base64"])
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[0] > 0


def test_no_violation_result_matches_contract(make_settings, video_path: Path) -> None:
    with TestClient(create_app(make_settings(stub_mode="none"))) as client:
        body = wait_for(client, submit(client, video_path).json()["job_id"])
    assert validate_result(body) == []
    assert body["verdict"] == "no_violation"
    assert body["violations"] == []


def test_corrupt_upload_marks_job_failed(client: TestClient, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mov"
    broken.write_bytes(os.urandom(4096))
    body = wait_for(client, submit(client, broken).json()["job_id"])
    assert body["status"] == "failed"
    assert body["job_id"]


def test_upload_over_limit_returns_413(make_settings, video_path: Path) -> None:
    with TestClient(create_app(make_settings(max_upload_mb=0.001))) as client:
        assert submit(client, video_path).status_code == 413
        assert list((client.app.state.settings.uploads_dir).glob("*")) == []


def test_upload_removed_after_processing(make_settings, video_path: Path) -> None:
    settings = make_settings()
    with TestClient(create_app(settings)) as client:
        wait_for(client, submit(client, video_path).json()["job_id"])
    assert list(settings.uploads_dir.glob("*")) == []


def test_results_survive_restart(make_settings, video_path: Path) -> None:
    settings = make_settings()
    with TestClient(create_app(settings)) as client:
        job_id = submit(client, video_path).json()["job_id"]
        first = wait_for(client, job_id)
    with TestClient(create_app(settings)) as client:
        again = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
    assert again.status_code == 200
    assert again.json() == first


class BlockingAnalyzer:
    name = "blocking"

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()

    def analyze(self, video_path, progress, deadline):
        self.started.set()
        progress(0.5)
        self.release.wait(10)
        return [Event(type_id="stub_motion", timecode_sec=1.0, confidence=0.9, observation="test", evidence=[EvidenceRef(1.0)])]


def test_processing_state_is_reported(make_settings, video_path: Path) -> None:
    analyzer = BlockingAnalyzer()
    with TestClient(create_app(make_settings(), analyzer=analyzer)) as client:
        first = submit(client, video_path).json()["job_id"]
        second = submit(client, video_path).json()["job_id"]
        assert analyzer.started.wait(5)
        for job_id in (first, second):
            body = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()
            assert body["status"] == "processing"
            assert body["job_id"] == job_id
        analyzer.release.set()
        assert validate_result(wait_for(client, first)) == []
        assert validate_result(wait_for(client, second)) == []


def test_api_extra_fields_are_opt_in(make_settings, video_path: Path) -> None:
    with TestClient(create_app(make_settings(api_extra_fields=True))) as client:
        body = wait_for(client, submit(client, video_path).json()["job_id"])
    assert validate_result(body, strict_keys=False) == []
    assert {"confidence", "rule"} <= body["violations"][0].keys()


def test_ui_endpoints_are_open_and_detailed(client: TestClient, video_path: Path) -> None:
    response = submit(client, video_path, name="clip.mp4", path="/ui/api/analyze", headers={})
    assert response.status_code == 200
    body = wait_for(client, response.json()["job_id"], path="/ui/api/jobs/{}", headers={})
    assert body["status"] == "done"
    assert {"confidence", "rule", "title"} <= body["violations"][0].keys()
    assert client.get("/").status_code == 200


def test_ui_can_be_disabled(make_settings, video_path: Path) -> None:
    with TestClient(create_app(make_settings(ui_enabled=False))) as client:
        assert submit(client, video_path, path="/ui/api/analyze", headers={}).status_code == 404


def test_validator_rejects_contract_breaks() -> None:
    frame = {"timestamp_sec": 1.0, "image_base64": base64.b64encode(b"\xff\xd8\xff\xe0data").decode()}
    good = {"job_id": "a", "status": "done", "verdict": "violation", "violations": [
        {"type": "t", "timecode_sec": 1.0, "description": "d", "evidence_frames": [frame]}]}
    assert validate_result(good) == []
    assert validate_result({**good, "verdict": "no_violation"})
    assert validate_result({**good, "violations": []})
    data_uri = {**frame, "image_base64": "data:image/jpeg;base64," + frame["image_base64"]}
    assert validate_result({**good, "violations": [{**good["violations"][0], "evidence_frames": [data_uri]}]})
    assert validate_result({**good, "violations": [{**good["violations"][0], "evidence_frames": [frame] * 4}]})
    assert validate_result({**good, "extra": 1})


@pytest.mark.skipif(not os.environ.get("CASHWATCH_SAMPLE_VIDEO"), reason="set CASHWATCH_SAMPLE_VIDEO to a real .mov")
def test_real_video_within_budget(client: TestClient) -> None:
    video = Path(os.environ["CASHWATCH_SAMPLE_VIDEO"])
    started = time.monotonic()
    body = wait_for(client, submit(client, video, name=video.name).json()["job_id"], timeout=300)
    assert validate_result(body) == []
    assert time.monotonic() - started < 300


def test_synthetic_writer_roundtrip(tmp_path: Path) -> None:
    path = write_video(tmp_path / "x.mov", seconds=1.0)
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 25
    cap.release()
