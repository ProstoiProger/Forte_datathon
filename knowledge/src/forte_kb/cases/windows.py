"""Clip windows over the decoded frames and their relation to the labeled events.

Relations:
  event   - the window contains a labeled violation second (positive)
  near    - within NEAR_SEC of a labeled second but not containing it (label resolution is 1 s, onset may differ)
  other   - elsewhere in a violation video: unlabeled, the violation may persist (a safe left open stays open)
  normal  - any window of a clean video (negative)
Only event and normal windows are used as supervised examples.
"""

NEAR_SEC = 3.0


def window_starts(duration: float, window: float, stride: float) -> list[float]:
    if duration <= window:
        return [0.0]
    starts = []
    s = 0.0
    while s + window <= duration + 1e-6:
        starts.append(round(s, 3))
        s += stride
    tail = round(duration - window, 3)
    if tail - starts[-1] > stride / 2:  # cover the end of the video
        starts.append(tail)
    return starts


def relation(start: float, end: float, events: list[dict], is_violation: bool) -> tuple[str, dict | None]:
    if not is_violation:
        return "normal", None
    inside = [e for e in events if start <= e["t"] <= end]
    if inside:
        mid = (start + end) / 2
        return "event", min(inside, key=lambda e: abs(e["t"] - mid))
    near = [e for e in events if start - NEAR_SEC <= e["t"] <= end + NEAR_SEC]
    if near:
        return "near", min(near, key=lambda e: min(abs(e["t"] - start), abs(e["t"] - end)))
    return "other", None


def frames_in(frames: list[dict], start: float, end: float) -> list[dict]:
    """Frames with start <= t < end (end inclusive only for the last frame of the video)."""
    sel = [f for f in frames if start - 1e-6 <= f["t"] < end - 1e-6]
    return sel or [min(frames, key=lambda f: abs(f["t"] - start))]


def build_windows(video: dict, frames: list[dict], duration: float, window: float, stride: float) -> list[dict]:
    is_violation = video["label"] == "violation"
    out = []
    for start in window_starts(duration, window, stride):
        end = round(min(start + window, duration), 3)
        rel, event = relation(start, end, video["events"], is_violation)
        sel = frames_in(frames, start, end)
        out.append({
            "id": f"{video['video']}@{start:06.2f}",
            "video": video["video"],
            "group": video["group"],
            "label": video["label"],
            "start": start,
            "end": end,
            "relation": rel,
            "event_id": event["event_id"] if event else None,
            "event_t": event["t"] if event else None,
            "event_type": event["type"] if event else None,
            "frames": [f["file"] for f in sel],
            "frame_times": [f["t"] for f in sel],
        })
    return out
