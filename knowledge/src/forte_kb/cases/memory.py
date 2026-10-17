"""Read-only access to the case memory for the analyzer at inference time.

Typical use for a new video: embed its clip windows with the same CLIP instruction, call similar_clips() per
window, and pass the returned cases (keyframes, what happened, candidate rules) to the VLM as reference context.
"""

import json
from pathlib import Path

import numpy as np

from ..store import Space


class CaseMemory:
    def __init__(self, root: Path):
        self.root = Path(root)
        with (self.root / "cases.jsonl").open(encoding="utf-8") as fh:
            self.cases = {c["video_id"]: c for c in (json.loads(line) for line in fh if line.strip())}
        self.events = {e["event_id"]: {**e, "video_id": c["video_id"], "group": c["group"]}
                       for c in self.cases.values() for e in c["events"]}
        vec = self.root / "vectors"
        self.spaces = {n: Space.load(vec, n) for n in ("clip", "frame", "card") if (vec / f"{n}.npy").exists()}

    def _where(self, exclude_videos: set[str] | None, exclude_groups: set[str] | None, relations: set[str] | None):
        def ok(m: dict) -> bool:
            return ((not exclude_videos or m["video"] not in exclude_videos)
                    and (not exclude_groups or m["group"] not in exclude_groups)
                    and (not relations or m["relation"] in relations))
        return ok

    def similar_clips(self, vector: np.ndarray, k: int = 5, exclude_videos=None, exclude_groups=None,
                      relations=("event", "normal")) -> list[dict]:
        """Nearest training windows; event hits carry the full event record (description, type, rules)."""
        hits = self.spaces["clip"].search(vector, k, self._where(exclude_videos, exclude_groups, set(relations)))
        out = []
        for h in hits:
            ev = self.events.get(h["event_id"]) if h["relation"] == "event" else None
            # window metadata keeps the original hypothesis; the event record carries the reviewed type
            out.append({**h, "event_type": ev["type"]["id"] if ev else h.get("event_type"), "event": ev})
        return out

    def similar_cards(self, vector: np.ndarray, k: int = 5, kinds=("event", "window"), **kw) -> list[dict]:
        where = self._where(kw.get("exclude_videos"), kw.get("exclude_groups"), None)
        return self.spaces["card"].search(vector, k, lambda m: m["kind"] in kinds and where(m))

    def suspiciousness(self, vectors: np.ndarray, k: int = 10, exclude_groups=None) -> np.ndarray:
        """Share of event windows among the k nearest labeled training windows, per query vector."""
        space = self.spaces["clip"]
        keep = [i for i, m in enumerate(space.meta) if m["relation"] in ("event", "normal")
                and (not exclude_groups or m["group"] not in exclude_groups)]
        y = np.array([space.meta[i]["relation"] == "event" for i in keep], dtype=float)
        sims = np.asarray(vectors, dtype=np.float32) @ space.vectors[keep].T
        k = min(k, len(keep))
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        return y[top].mean(axis=1)
