from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from cashwatch.video import VideoError


def _flip(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)


def _crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    y0, x0 = int(h * 0.05), int(w * 0.12)
    y1, x1 = int(h * 0.85), int(w * 0.97)
    return cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)


def _warp(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.10, h * 0.04], [w * 0.97, 0], [w, h], [0, h * 0.94]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _rotate(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 8.0, 1.12)
    return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _dark(frame: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(frame, alpha=0.65, beta=-10)


def _lowres(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "flip": _flip,
    "crop": _crop,
    "warp": _warp,
    "rotate": _rotate,
    "dark": _dark,
    "lowres": _lowres,
}


def apply(frame: np.ndarray, name: str) -> np.ndarray:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}; choose from {', '.join(TRANSFORMS)}")
    return TRANSFORMS[name](frame)


def transform_video(src: Path, dst: Path, name: str) -> Path:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}; choose from {', '.join(TRANSFORMS)}")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise VideoError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial" + dst.suffix)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise VideoError(f"cannot write {tmp}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(TRANSFORMS[name](frame))
    finally:
        cap.release()
        writer.release()
    tmp.replace(dst)
    return dst
