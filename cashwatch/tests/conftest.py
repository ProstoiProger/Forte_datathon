import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cashwatch.config import Settings
from cashwatch.main import create_app

API_KEY = "test-key-0123456789"
HEADERS = {"X-API-Key": API_KEY}


def write_video(path: Path, seconds: float = 4.0, fps: float = 30.0, size: tuple[int, int] = (160, 288)) -> Path:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    total = int(seconds * fps)
    for i in range(total):
        frame = np.full((height, width, 3), 200, np.uint8)
        t = i / fps
        if 2.0 <= t < 3.0:
            x = int((t - 2.0) * (width - 40))
            cv2.rectangle(frame, (x, 100), (x + 40, 160), (20, 20, 20), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_video(tmp_path_factory.mktemp("videos") / "sample.mov")


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    def factory(**overrides) -> Settings:
        values = {"api_keys": API_KEY, "data_dir": tmp_path / "var", "analyzer": "stub", "stub_mode": "peak"}
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def client(make_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    with TestClient(create_app(make_settings())) as c:
        yield c


def wait_for(client: TestClient, job_id: str, path: str = "/api/v1/jobs/{}", headers: dict | None = None, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(path.format(job_id), headers=HEADERS if headers is None else headers)
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] != "processing":
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def submit(client: TestClient, video: Path, name: str = "video_017.mov", path: str = "/api/v1/analyze", headers: dict | None = None):
    with video.open("rb") as fh:
        return client.post(path, headers=HEADERS if headers is None else headers, files={"file": (name, fh, "video/quicktime")})
