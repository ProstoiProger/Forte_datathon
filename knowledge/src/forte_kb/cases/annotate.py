"""VLM annotation passes over the training videos (local Ollama model, resumable).

facts     every non-overlapping 4 s window: per-frame state of the concepts that the model recognized reliably in
          the frame bake-off (Datathon/runs/vlm_bakeoff), plus visible actions and a one-line scene description
events    every labeled violation: what happens before / at / after the labeled second, the visible cue that
          separates it from normal work, and a type hypothesis from the catalog (or a new type)
overview  every video: room layout, people and the order of actions over the whole video

The events pass is label-conditioned (the model is told where the organizers marked a violation), so its output
is a hypothesis for human review, never ground truth. All passes ask for visible facts only and roles, not faces.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from ..vlm import Ollama

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
ROLES = ["cashier", "client", "collector", "other_person", "unknown"]
CONCEPTS = {
    "people": "integer: persons inside the room",
    "cashier_at_workplace": "the cashier sits or stands at the cashier desk",
    "cashier_leaving": "the cashier walks away from the desk or is at/through the door",
    "cash_on_desk": "banknotes or cash bundles lie on the desk",
    "cash_in_hands": "the cashier holds banknotes",
    "safe_open": "the door of the metal safe/cabinet is open",
    "door_open": "the room door is open",
    "phone_in_hand": "the cashier holds a mobile phone",
    "drinking": "the cashier holds a cup or drink, or drinks",
    "personal_items_on_desk": "a coat, handbag or other personal bag is on the desk",
    "collection_bag": "a cash-collection bag is visible",
    "visitor_present": "any person other than the cashier is in the room",
}

SYSTEM = (
    "You annotate footage from a fixed top-down CCTV camera in a bank cash room for an audit database. "
    "Report only what is visible in the frames. If something is not visible or unclear, say so instead of guessing. "
    "Refer to people only by role (cashier, client, collector, other person); never describe or identify faces."
)


def _frame_list(times: list[float]) -> str:
    return "\n".join(f"Frame {i + 1}: t={t:.1f}s" for i, t in enumerate(times))


FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {"type": "array", "items": {
            "type": "object",
            "properties": {"frame": {"type": "integer"},
                           **{k: {"type": "integer" if k == "people" else "boolean"} for k in CONCEPTS}},
            "required": ["frame", *CONCEPTS]}},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {"actor": {"type": "string", "enum": ROLES}, "action": {"type": "string"},
                           "object": {"type": "string"}, "from_frame": {"type": "integer"},
                           "to_frame": {"type": "integer"}},
            "required": ["actor", "action", "object", "from_frame", "to_frame"]}},
        "scene_ru": {"type": "string"},
    },
    "required": ["frames", "actions", "scene_ru"],
}

FACTS_PROMPT = """These {n} frames are in time order from one fixed top-down camera in a bank cash room:
{frames}

1. For EACH frame give its state with these keys:
{concepts}
2. List the visible actions in this interval. "action" is a short snake_case verb phrase (for example
opens_safe, hands_bag_to_cashier, walks_out_of_door, counts_banknotes, sits_down_at_desk); "object" is what the
action involves or an empty string; from_frame / to_frame are frame numbers. A client usually sits across the desk;
a collector usually carries a cash-collection bag.
3. scene_ru: one sentence in Russian describing what happens in this interval.
Answer only with JSON."""

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "layout_ru": {"type": "string"},
        "people_ru": {"type": "string"},
        "procedure_ru": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["layout_ru", "people_ru", "procedure_ru", "objects"],
}

OVERVIEW_PROMPT = """These {n} frames are spread evenly over a {duration:.0f}-second recording from one fixed top-down
camera in a bank cash room, in time order:
{frames}

Answer in Russian:
- layout_ru: where in the frame the cashier desk, the safe, the door and other fixed equipment are
  (use frame directions: top, bottom, left, right, centre);
- people_ru: which roles appear (cashier, client, collector, other person) and roughly when;
- procedure_ru: what happens over the recording, in time order, with approximate seconds;
- objects: short English names of the movable objects that appear (banknotes, collection bag, phone, cup, ...).
Answer only with JSON."""


def event_schema(type_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "before_ru": {"type": "string"},
            "at_label_ru": {"type": "string"},
            "after_ru": {"type": "string"},
            "visual_events": {"type": "array", "items": {
                "type": "object",
                "properties": {"t": {"type": "number"}, "actor": {"type": "string", "enum": ROLES},
                               "action": {"type": "string"}, "object": {"type": "string"}},
                "required": ["t", "actor", "action", "object"]}},
            "violation_type": {"type": "string", "enum": [*type_ids, "new", "not_visible"]},
            "new_type_title_ru": {"type": "string"},
            "distinguishing_cue_ru": {"type": "string"},
            "evidence_frame": {"type": "integer"},
            "confidence": {"type": "number"},
        },
        "required": ["before_ru", "at_label_ru", "after_ru", "visual_events", "violation_type",
                     "new_type_title_ru", "distinguishing_cue_ru", "evidence_frame", "confidence"],
    }


EVENT_PROMPT = """These {n} frames are in time order from one fixed top-down camera in a bank cash room:
{frames}

A human reviewer marked a possible violation of cash-handling discipline at about t={t:.0f}s (to within a second).
Describe objectively, in Russian, what happens before (before_ru), at (at_label_ru) and after (after_ru) that moment.
List the visible actions as visual_events (t in seconds, actor role, snake_case action, object).
Choose the violation type that the visible facts support from this catalog:
{catalog}
Use "new" with a short Russian title in new_type_title_ru if the visible situation is a different breach, and
"not_visible" if you cannot see any breach in these frames. distinguishing_cue_ru: the visible detail that separates
this moment from normal work in the same room. evidence_frame: the frame number that best shows it.
confidence: 0..1 for the chosen type. Answer only with JSON."""


def nearest_frames(frames: list[dict], times: list[float]) -> list[dict]:
    return [min(frames, key=lambda f: abs(f["t"] - t)) for t in times]


def facts_jobs(video: dict, frames: list[dict], duration: float, window: float = 4.0, per_window: int = 4) -> Iterator[dict]:
    step = window / per_window
    start = 0.0
    while start < duration - step / 2:
        times = [start + step * (i + 0.5) for i in range(per_window) if start + step * (i + 0.5) < duration]
        sel = nearest_frames(frames, times)
        yield {"key": f"{video['video']}@{start:06.2f}", "video": video["video"], "start": start,
               "end": min(start + window, duration), "frames": sel}
        start += window


def event_jobs(video: dict, frames: list[dict], duration: float, half: int = 4) -> Iterator[dict]:
    for e in video["events"]:
        times = [t for t in (e["t"] + d for d in range(-half, half + 1)) if 0 <= t <= duration]
        yield {"key": e["event_id"], "video": video["video"], "event": e, "frames": nearest_frames(frames, times)}


def overview_job(video: dict, frames: list[dict], duration: float, n: int = 12) -> dict:
    times = [duration * (i + 0.5) / n for i in range(n)]
    return {"key": video["video"], "video": video["video"], "duration": duration,
            "frames": nearest_frames(frames, times)}


def catalog_text(types: dict) -> str:
    return "\n".join(f"- {tid}: {t['title']} ({t['rationale']})" for tid, t in types.items())


class Annotator:
    def __init__(self, vlm: Ollama, frames_root: Path, out_dir: Path, types: dict):
        self.vlm = vlm
        self.frames_root = frames_root
        self.out_dir = out_dir
        self.types = types
        out_dir.mkdir(parents=True, exist_ok=True)

    def _done(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        done = set()
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("parsed") is not None and r.get("prompt_version") == PROMPT_VERSION:
                done.add(r["key"])
        return done

    def _paths(self, job: dict) -> list[Path]:
        return [self.frames_root / job["video"] / f["file"] for f in job["frames"]]

    def _run(self, name: str, jobs: list[dict], make) -> None:
        path = self.out_dir / f"{name}.jsonl"
        done = self._done(path)
        todo = [j for j in jobs if j["key"] not in done]
        log.info("%s: %d jobs, %d cached, %d to run", name, len(jobs), len(jobs) - len(todo), len(todo))
        with path.open("a", encoding="utf-8") as fh:
            for i, job in enumerate(todo, 1):
                prompt, schema = make(job)
                parsed, raw, sec = self.vlm.chat_json(prompt, self._paths(job), schema, system=SYSTEM)
                rec = {"key": job["key"], "video": job["video"], "model": self.vlm.model,
                       "prompt_version": PROMPT_VERSION, "frame_times": [f["t"] for f in job["frames"]],
                       "seconds": round(sec, 2), "parsed": parsed, "raw": None if parsed is not None else raw}
                rec.update({k: job[k] for k in ("start", "end") if k in job})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                log.info("%s %d/%d %s %.1fs%s", name, i, len(todo), job["key"], sec, "" if parsed else " PARSE FAIL")

    def facts(self, jobs: list[dict]) -> None:
        concepts = "\n".join(f'- "{k}": {v}' for k, v in CONCEPTS.items())

        def make(job):
            times = [f["t"] for f in job["frames"]]
            return FACTS_PROMPT.format(n=len(times), frames=_frame_list(times), concepts=concepts), FACTS_SCHEMA
        self._run("facts", jobs, make)

    def events(self, jobs: list[dict]) -> None:
        schema = event_schema(list(self.types))
        catalog = catalog_text(self.types)

        def make(job):
            times = [f["t"] for f in job["frames"]]
            return EVENT_PROMPT.format(n=len(times), frames=_frame_list(times), t=job["event"]["t"],
                                       catalog=catalog), schema
        self._run("events", jobs, make)

    def overview(self, jobs: list[dict]) -> None:
        def make(job):
            times = [f["t"] for f in job["frames"]]
            return OVERVIEW_PROMPT.format(n=len(times), duration=job["duration"],
                                          frames=_frame_list(times)), OVERVIEW_SCHEMA
        self._run("overview", jobs, make)


def load_pass(out_dir: Path, name: str) -> dict[str, dict]:
    """Latest successful record per key for the current prompt version."""
    path = out_dir / f"{name}.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("parsed") is not None and r.get("prompt_version") == PROMPT_VERSION:
                out[r["key"]] = r
    return out
