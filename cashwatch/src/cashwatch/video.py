import base64
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoError(Exception):
    pass


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError("file is not a readable video")
    return cap


def probe(path: Path) -> VideoInfo:
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None or fps <= 0 or count <= 0:
        raise VideoError("video has no decodable frames")
    return VideoInfo(fps=fps, frame_count=count, duration_sec=count / fps, width=width, height=height)


def iter_frames(path: Path, fps: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    cap = _open(path)
    step = 1.0 / fps if fps else 0.0
    next_t = 0.0
    try:
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if step:
                next_t = (math.floor(t / step + 1e-6) + 1) * step
            yield t, frame
    finally:
        cap.release()


def frames_at(path: Path, times: Sequence[float]) -> list[tuple[float, np.ndarray]]:
    if not times:
        return []
    order = sorted(range(len(times)), key=lambda i: times[i])
    found: dict[int, tuple[float, np.ndarray]] = {}
    pending = list(order)
    prev: tuple[float, np.ndarray] | None = None
    for t, frame in iter_frames(path):
        while pending and t >= times[pending[0]]:
            idx = pending.pop(0)
            target = times[idx]
            if prev is not None and abs(prev[0] - target) < abs(t - target):
                found[idx] = prev
            else:
                found[idx] = (t, frame)
        if not pending:
            break
        prev = (t, frame)
    for idx in pending:
        if prev is not None:
            found[idx] = prev
    return [found[i] for i in range(len(times)) if i in found]


def draw_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    out = frame.copy()
    cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 2)
    return out


def encode_jpeg_b64(frame: np.ndarray, quality: int = 85, max_side: int = 960) -> str:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("jpeg encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
