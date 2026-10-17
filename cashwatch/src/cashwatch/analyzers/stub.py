import time
from pathlib import Path

import cv2
import numpy as np

from cashwatch.analyzers.base import Event, EvidenceRef, ProgressFn
from cashwatch.video import iter_frames, probe

SAMPLE_FPS = 4.0
MIN_MOTION = 0.5


class MotionStubAnalyzer:
    name = "stub"
    default_threshold = 0.0

    def __init__(self, mode: str = "peak") -> None:
        self.mode = mode

    def analyze(self, video_path: Path, progress: ProgressFn, deadline: float) -> list[Event]:
        info = probe(video_path)
        prev: np.ndarray | None = None
        best_energy, best_t = 0.0, None
        for t, frame in iter_frames(video_path, fps=SAMPLE_FPS):
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, max(1, 64 * info.height // max(1, info.width))))
            small = small.astype(np.float32)
            if prev is not None:
                energy = float(np.mean(np.abs(small - prev)))
                if energy > best_energy:
                    best_energy, best_t = energy, t
            prev = small
            progress(min(1.0, t / info.duration_sec))
            if time.monotonic() > deadline:
                break
        if self.mode == "none" or best_t is None or best_energy < MIN_MOTION:
            return []
        times = [max(0.0, best_t - 0.5), best_t, min(info.duration_sec, best_t + 0.5)]
        return [
            Event(
                type_id="stub_motion",
                timecode_sec=best_t,
                confidence=min(1.0, best_energy / 20.0),
                observation=f"Наибольшее изменение изображения около {best_t:.1f} с.",
                evidence=[EvidenceRef(t) for t in times],
            )
        ]
