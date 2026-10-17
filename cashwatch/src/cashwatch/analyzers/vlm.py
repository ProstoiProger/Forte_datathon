import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cashwatch.analyzers.base import AnalyzerError, Event, EvidenceRef, ProgressFn
from cashwatch.rules import Registry, ViolationType, load_registry
from cashwatch.video import encode_jpeg_b64, iter_frames, probe

log = logging.getLogger("cashwatch.vlm")

NORMAL = "normal"
NORMAL_TITLE = "пример обычной работы без нарушения"

SYSTEM_PROMPT = (
    "You review frames from a fixed CCTV camera in a bank cash room. A cashier works at a desk with a cash counting "
    "machine, a computer and a metal safe. Clients and cash collectors may enter through a glass door. "
    "Your job is to find cash-handling discipline violations that are directly visible in the frames. "
    "Report only what can be seen. Do not guess about documents, amounts, accounts or anything outside the frames. "
    "Do not identify people; describe them only by role, position and clothing. "
    "Normal work is not a violation: serving a client at the desk, counting cash in front of the client, "
    "using the computer, putting cash into the safe and locking it."
)

FRAMES_ONLY = "Frames ({n}) in time order:\n{frames}"

WITH_REFERENCES = """Reference images 1-{r} are examples from other recordings. Use them only to understand what each type looks like; never report events from them:
{references}

Frames to analyze are images {first}-{last}, in time order:
{frames}"""

USER_TEMPLATE = """{images}

Violation types to check:
{types}

Answer in JSON with the key "events". For each violation you can see in the frames to analyze add one item:
- "type_id": one of the ids above,
- "frame": number of the frame (1-{n}) where the violation starts or is most clearly visible,
- "confidence": number from 0 to 1,
- "observation": one or two sentences in Russian describing only what is visible.
If a violation continues over several frames, report it once, at the frame where it starts.
Return {{"events": []}} if none of the frames shows a violation."""


class VlmCallError(Exception):
    pass


@dataclass(frozen=True)
class Frame:
    t: float
    image: str


@dataclass(frozen=True)
class Reference:
    type_id: str
    image: str
    source: str = ""
    group: str = ""


class OllamaClient:
    def __init__(self, url: str, model: str, timeout: float = 90.0, num_ctx: int = 16384, keep_alive: str = "60m") -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}{path}", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise VlmCallError(f"{path}: {exc}") from exc

    def chat(self, system: str, user: str, images: list[str], schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": images},
            ],
        }
        data = self._post("/api/chat", payload)
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise VlmCallError(f"no message content: {str(data)[:200]}")
        return content

    def check(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=10) as response:
                names = [m.get("name", "") for m in json.loads(response.read().decode("utf-8")).get("models", [])]
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return False, f"ollama unreachable at {self.url}: {exc}"
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"model {self.model} not pulled; available: {', '.join(names) or 'none'}"
        return True, f"ollama {self.model}"


def sample_frames(path: Path, fps: float, max_side: int, quality: int) -> list[Frame]:
    frames = []
    for t, image in iter_frames(path, fps=fps):
        frames.append(Frame(t=round(t, 3), image=encode_jpeg_b64(image, quality=quality, max_side=max_side)))
    return frames


def plan_windows(duration: float, window: float, stride: float, max_windows: int) -> tuple[float, float]:
    if max_windows < 2 or duration <= window:
        return window, stride
    needed = math.ceil((duration - window) / stride) + 1
    if needed <= max_windows:
        return window, stride
    wider = (duration - window) / (max_windows - 1)
    if wider <= window:
        return window, wider
    span = duration / max_windows
    return span, span


def make_windows(times: list[float], duration: float, window: float, stride: float, max_frames: int) -> list[list[int]]:
    windows = []
    start = 0.0
    while True:
        idx = [i for i, t in enumerate(times) if start <= t < start + window]
        if len(idx) > max_frames:
            step = (len(idx) - 1) / (max_frames - 1)
            idx = [idx[round(k * step)] for k in range(max_frames)]
        if idx:
            windows.append(idx)
        if start + window >= duration - 1e-9 or stride <= 0:
            break
        start += stride
    return windows


def response_schema(type_ids: list[str], n_frames: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type_id": {"type": "string", "enum": type_ids},
                        "frame": {"type": "integer", "minimum": 1, "maximum": n_frames},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "observation": {"type": "string"},
                    },
                    "required": ["type_id", "frame", "confidence", "observation"],
                },
            }
        },
        "required": ["events"],
    }


def build_user_prompt(frames: list[Frame], types: list[ViolationType], references: list[Reference] | None = None) -> str:
    refs = references or []
    titles = {t.type_id: t.title for t in types}
    if refs:
        offset = len(refs)
        frame_lines = "\n".join(f"Frame {i} (image {offset + i}): t={f.t:.1f}s" for i, f in enumerate(frames, 1))
        ref_lines = "\n".join(
            f"Reference image {i}: {r.type_id} - {NORMAL_TITLE if r.type_id == NORMAL else titles.get(r.type_id, r.type_id)}"
            for i, r in enumerate(refs, 1)
        )
        images = WITH_REFERENCES.format(r=len(refs), references=ref_lines, first=offset + 1, last=offset + len(frames), frames=frame_lines)
    else:
        frame_lines = "\n".join(f"Frame {i}: t={f.t:.1f}s" for i, f in enumerate(frames, 1))
        images = FRAMES_ONLY.format(n=len(frames), frames=frame_lines)
    type_lines = "\n".join(f"- {t.type_id}: {t.title}. {t.rationale}" for t in types)
    return USER_TEMPLATE.format(images=images, types=type_lines, n=len(frames))


def parse_response(content: str, frames: list[Frame], valid: set[str], window: int = -1) -> list[Event]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("model returned non-JSON content: %s", content[:200])
        return []
    items = data.get("events") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    events = []
    for item in items:
        if not isinstance(item, dict) or item.get("type_id") not in valid:
            continue
        try:
            frame = int(item.get("frame"))
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        if not 1 <= frame <= len(frames):
            continue
        k = frame - 1
        observation = str(item.get("observation") or "").strip()[:400]
        evidence = [EvidenceRef(frames[j].t) for j in (k - 1, k, k + 1) if 0 <= j < len(frames)]
        events.append(
            Event(
                type_id=item["type_id"],
                timecode_sec=frames[k].t,
                confidence=max(0.0, min(1.0, confidence)),
                observation=observation,
                evidence=evidence,
                window=window,
            )
        )
    return events


def cluster_events(events: list[Event], gap: float) -> list[list[Event]]:
    clusters: list[list[Event]] = []
    for type_id in sorted({e.type_id for e in events}):
        current: list[Event] = []
        for e in sorted((e for e in events if e.type_id == type_id), key=lambda e: e.timecode_sec):
            if current and e.timecode_sec - current[-1].timecode_sec > gap:
                clusters.append(current)
                current = []
            current.append(e)
        if current:
            clusters.append(current)
    return clusters


def _best(cluster: list[Event]) -> Event:
    return max(cluster, key=lambda x: (x.confidence, -x.timecode_sec))


def merge_events(events: list[Event], gap: float) -> list[Event]:
    return sorted((_best(c) for c in cluster_events(events, gap)), key=lambda e: e.timecode_sec)


def score_events(clusters: list[list[Event]], spans: list[tuple[float, float]], mode: str = "agreement") -> list[Event]:
    scored = []
    for cluster in clusters:
        best = _best(cluster)
        votes = len({e.window for e in cluster})
        coverage = max(1, sum(1 for a, b in spans if a - 0.01 <= best.timecode_sec <= b + 0.01))
        support = min(1.0, votes / coverage)
        score = best.confidence * (0.5 + 0.5 * support) if mode == "agreement" else best.confidence
        scored.append(replace(best, confidence=round(score, 4), votes=votes, coverage=coverage, raw_confidence=best.confidence))
    return sorted(scored, key=lambda e: e.timecode_sec)


def load_references(directory: Path, max_side: int = 512) -> list[Reference]:
    manifest = directory / "manifest.json"
    if manifest.exists():
        items = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        items = [{"type_id": p.stem.split("__")[0], "file": p.name} for p in sorted(directory.glob("*.jpg"))]
    refs = []
    for item in items:
        path = directory / item["file"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("reference %s is not a readable image", path)
            continue
        refs.append(Reference(item["type_id"], encode_jpeg_b64(image, quality=85, max_side=max_side), item.get("source_video", ""), item.get("group", "")))
    return refs


class VlmAnalyzer:
    name = "vlm"
    default_threshold = 0.5

    def __init__(
        self,
        client: Any,
        registry: Registry | None = None,
        fps: float = 2.0,
        window_sec: float = 6.0,
        stride_sec: float = 3.0,
        max_frames: int = 12,
        max_side: int = 848,
        jpeg_quality: int = 85,
        retries: int = 1,
        merge_gap_sec: float = 4.0,
        safety_sec: float = 10.0,
        max_windows: int = 20,
        score_mode: str = "agreement",
        references: list[Reference] | None = None,
        max_references: int = 6,
    ) -> None:
        self.client = client
        registry = registry or load_registry()
        self.types = [t for t in registry.all() if t.verified_citations]
        self.valid = {t.type_id for t in self.types}
        self.fps = fps
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.max_frames = max_frames
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.retries = retries
        self.merge_gap_sec = merge_gap_sec
        self.safety_sec = safety_sec
        self.max_windows = max_windows
        self.score_mode = score_mode
        self.references = [r for r in references or [] if r.type_id in self.valid or r.type_id == NORMAL]
        self.max_references = max_references
        self.exclude_reference_groups: set[str] = set()
        self.last_stats: dict[str, Any] = {}

    def check(self) -> tuple[bool, str]:
        check = getattr(self.client, "check", None)
        return check() if check else (True, "client without check")

    def active_references(self) -> list[Reference]:
        refs = [r for r in self.references if not (r.group and r.group in self.exclude_reference_groups)]
        return refs[: self.max_references]

    def _call(self, frames: list[Frame], refs: list[Reference]) -> str | None:
        schema = response_schema(sorted(self.valid), len(frames))
        prompt = build_user_prompt(frames, self.types, refs)
        images = [r.image for r in refs] + [f.image for f in frames]
        for attempt in range(self.retries + 1):
            try:
                return self.client.chat(SYSTEM_PROMPT, prompt, images, schema)
            except VlmCallError as exc:
                log.warning("vlm call failed (attempt %d): %s", attempt + 1, exc)
        return None

    def analyze(self, video_path: Path, progress: ProgressFn, deadline: float) -> list[Event]:
        info = probe(video_path)
        frames = sample_frames(video_path, self.fps, self.max_side, self.jpeg_quality)
        window, stride = plan_windows(info.duration_sec, self.window_sec, self.stride_sec, self.max_windows)
        windows = make_windows([f.t for f in frames], info.duration_sec, window, stride, self.max_frames)
        refs = self.active_references()
        raw: list[Event] = []
        spans: list[tuple[float, float]] = []
        attempted = failed = 0
        started = time.monotonic()
        for i, idx in enumerate(windows):
            if time.monotonic() > deadline - self.safety_sec:
                log.warning("deadline reached after %d of %d windows", attempted, len(windows))
                break
            attempted += 1
            window_frames = [frames[j] for j in idx]
            content = self._call(window_frames, refs)
            if content is None:
                failed += 1
            else:
                spans.append((window_frames[0].t, window_frames[-1].t))
                raw.extend(parse_response(content, window_frames, self.valid, window=i))
            progress((i + 1) / len(windows))
        self.last_stats = {
            "frames": len(frames),
            "windows": len(windows),
            "window_sec": round(window, 2),
            "stride_sec": round(stride, 2),
            "references": len(refs),
            "attempted": attempted,
            "failed": failed,
            "raw_events": len(raw),
            "seconds": round(time.monotonic() - started, 2),
        }
        if attempted and failed == attempted:
            raise AnalyzerError(f"all {attempted} model calls failed")
        return score_events(cluster_events(raw, self.merge_gap_sec), spans, self.score_mode)

