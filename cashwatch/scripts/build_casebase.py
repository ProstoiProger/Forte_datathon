"""Build the case-base bundle (casebase.json + images/) from the knowledge store of the labeled training videos.

The bundle contains frames of the dataset videos: keep it under the git-ignored data/ folder, never commit it.
Inputs are read-only. With --eval the bundle is replayed leave-one-setup-out (see cashwatch.casebase.evaluate).
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from cashwatch.analyzers.facts.schema import FLAG_KEYS
from cashwatch.casebase import BUNDLE_FILE
from cashwatch.rules import load_registry

log = logging.getLogger("cashwatch.casebase")

CASES = Path(__file__).resolve().parents[2] / "knowledge" / "store" / "cases"
REVIEW = CASES.parents[1] / "cases_review.json"
NEAR_SEC = 3.0  # same window relations as the knowledge store
NEW_TYPE = "new"
FRAME_NAME = re.compile(r"t(\d{7})\.jpg$")
TIMES = re.compile(r"\s*\((?:с\s*)?[\d.,\s\-–]+с?\)")  # "(52-59 с)": clip times, meaningless in a reference


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def latest_facts(path: Path) -> dict[str, dict]:
    """The last record per window key that has parsed facts."""
    out: dict[str, dict] = {}
    for rec in read_jsonl(path):
        if rec.get("parsed") is None:
            continue
        out[rec["key"]] = rec
    return out


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


def provenance(records: list[dict]) -> dict:
    """Which perception produced the stored facts: runtime facts are comparable only with the same setup."""
    def most_common(values: list) -> object:
        return Counter(values).most_common(1)[0][0] if values else None

    models = Counter(r.get("model") for r in records)
    prompts = Counter(r.get("prompt_version") for r in records)
    out = {
        "model": most_common([r.get("model") for r in records]),
        "prompt_version": most_common([r.get("prompt_version") for r in records]),
        "window_sec": most_common([round(float(r["end"]) - float(r["start"]), 2) for r in records]),
        "frames_per_window": most_common([len(r.get("frame_times") or []) for r in records]),
    }
    if len(models) > 1 or len(prompts) > 1:
        out["mixed"] = {"models": dict(models), "prompt_versions": dict(prompts)}
    return out


def window_facts(parsed: dict) -> tuple[dict[str, bool], int]:
    frames = parsed.get("frames") or []
    flags = {k: any(bool(f.get(k)) for f in frames) for k in FLAG_KEYS}
    people = max((int(f.get("people") or 0) for f in frames), default=0)
    return flags, people


def frame_index(frames_dir: Path, video: str) -> list[tuple[float, Path]]:
    folder = frames_dir / video
    if not folder.is_dir():
        return []
    out = []
    for p in folder.iterdir():
        m = FRAME_NAME.search(p.name)
        if m:
            out.append((int(m.group(1)) / 1000.0, p))
    return sorted(out)


def save_image(frames: list[tuple[float, Path]], t: float, video: str, out: Path, max_side: int, quality: int,
               written: set[str]) -> str | None:
    """Frame nearest to t, resized and re-encoded once into images/; returns the bundle-relative path."""
    if not frames:
        return None
    ft, src = min(frames, key=lambda f: abs(f[0] - t))
    name = f"images/{video}_{round(ft * 1000):07d}.jpg"
    if name in written:
        return name
    image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        log.warning("cannot read %s", src)
        return None
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cannot encode {src}")
    (out / name).write_bytes(buf.tobytes())
    written.add(name)
    return name


def event_window(windows: list[dict], event_id: str, t: float, evidence_t: float | None) -> dict | None:
    """The stored window of an event: contains t, prefers the one that also contains the evidence time."""
    inside = [w for w in windows if w["start"] <= t <= w["end"]]
    if not inside:
        return None
    if evidence_t is not None:
        with_evidence = [w for w in inside if w["start"] <= evidence_t <= w["end"]]
        inside = with_evidence or inside
    own = [w for w in inside if w["event_id"] == event_id]
    return min(own or inside, key=lambda w: abs((w["start"] + w["end"]) / 2 - t))


def describe_event(event: dict, notes: str) -> str:
    """VLM description of the labeled moment; the reviewer's notes when the VLM saw another violation type."""
    vlm = event.get("vlm") or {}
    text = " ".join(s.strip() for s in (vlm.get("at_label_ru"), vlm.get("distinguishing_cue_ru")) if s)
    if (event.get("type") or {}).get("vlm_agrees") is False and notes:
        title = (event.get("type") or {}).get("title", "").strip()
        text = f"{title}. {TIMES.sub('', notes).strip()}" if title else TIMES.sub("", notes).strip()
    return text


def reviewed_type(event: dict, registry) -> str | None:
    type_id = (event.get("type") or {}).get("id")
    if not type_id or type_id == NEW_TYPE:
        return None
    if type_id not in registry:
        log.warning("%s: type %r is not in the registry, stored as untyped", event["event_id"], type_id)
        return None
    return type_id


def build(labels_path: Path, cases_path: Path, facts_path: Path, frames_dir: Path, review_path: Path | None,
          out: Path, max_side: int = 512, quality: int = 85) -> dict:
    registry = load_registry()
    labels = {v["video"]: v for v in json.loads(labels_path.read_text(encoding="utf-8"))["videos"]}
    cases = {c["video_id"]: c for c in read_jsonl(cases_path)}
    review = {}
    if review_path is not None and review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8")).get("events", {})
    case_events = {e["event_id"]: e for c in cases.values() for e in c.get("events", [])}
    facts = latest_facts(facts_path)

    (out / "images").mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    windows: list[dict] = []
    by_video: dict[str, list[dict]] = {}
    for rec in sorted(facts.values(), key=lambda r: (r["video"], r["start"])):
        video = labels.get(rec["video"])
        if video is None:
            log.warning("facts for unknown video %s skipped", rec["video"])
            continue
        frames = by_video.setdefault(rec["video"], frame_index(frames_dir, rec["video"]))
        start, end = float(rec["start"]), float(rec["end"])
        rel, ev = relation(start, end, video["events"], video["label"] == "violation")
        flags, people = window_facts(rec["parsed"])
        event_id = ev["event_id"] if ev else None
        windows.append({
            "id": rec["key"], "video": rec["video"], "group": video["group"], "label": video["label"],
            "start": start, "end": end, "relation": rel, "event_id": event_id,
            "type_id": reviewed_type(case_events[event_id], registry) if event_id in case_events else None,
            "flags": flags, "people": people, "scene_ru": (rec["parsed"].get("scene_ru") or "").strip(),
            "image": save_image(frames, (start + end) / 2, rec["video"], out, max_side, quality, written),
            "frame_times": [float(t) for t in rec.get("frame_times") or []],
        })

    events: list[dict] = []
    for vid, video in labels.items():
        frames = by_video.setdefault(vid, frame_index(frames_dir, vid))
        own_windows = [w for w in windows if w["video"] == vid]
        for label_event in video["events"]:
            e = case_events.get(label_event["event_id"])
            if e is None:
                log.warning("%s: no case record, skipped", label_event["event_id"])
                continue
            vlm = e.get("vlm") or {}
            evidence_t = vlm.get("evidence_t")
            t = float(e["t"])
            win = event_window(own_windows, e["event_id"], t, evidence_t)
            notes = (review.get(e["event_id"]) or {}).get("notes") or (e.get("review") or {}).get("notes", "")
            events.append({
                "event_id": e["event_id"], "video": vid, "group": video["group"], "t": t,
                "type_id": reviewed_type(e, registry), "title": (e.get("type") or {}).get("title", ""),
                "description_ru": describe_event(e, notes),
                "vlm_description_ru": " ".join(
                    s.strip() for s in (vlm.get("at_label_ru"), vlm.get("distinguishing_cue_ru")) if s),
                "vlm_agrees": (e.get("type") or {}).get("vlm_agrees"),
                "notes_ru": notes,
                "image": save_image(frames, float(evidence_t if evidence_t is not None else t), vid, out, max_side,
                                    quality, written),
                "window": win["id"] if win else None,
                "flags": win["flags"] if win else {k: False for k in FLAG_KEYS},
                "people": win["people"] if win else 0,
            })

    for stale in (out / "images").glob("*.jpg"):
        if f"images/{stale.name}" not in written:
            stale.unlink()
    relations: dict[str, int] = {}
    for w in windows:
        relations[w["relation"]] = relations.get(w["relation"], 0) + 1
    header = {
        "version": 1,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "contains frames of the dataset videos; never commit",
        "sources": {"labels": str(labels_path), "cases": str(cases_path), "facts": str(facts_path),
                    "frames": str(frames_dir), "review": str(review_path) if review_path else None},
        "counts": {"videos": len({w["video"] for w in windows}), "groups": len({w["group"] for w in windows}),
                   "windows": len(windows), "events": len(events), "images": len(written), "relations": relations},
        "facts": provenance([r for r in facts.values() if r["video"] in labels]),
        "image": {"max_side": max_side, "quality": quality},
        "flag_keys": list(FLAG_KEYS),
        "near_sec": NEAR_SEC,
    }
    videos = {vid: {"group": v["group"], "label": v["label"], "duration": (v.get("media") or {}).get("duration")}
              for vid, v in labels.items()}
    bundle = {"header": header, "videos": videos, "windows": windows, "events": events}
    (out / BUNDLE_FILE).write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    return header


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the empirical case base bundle from the knowledge store")
    parser.add_argument("--cases-dir", type=Path, default=CASES, help="knowledge store of the labeled videos")
    parser.add_argument("--labels", type=Path, help="default: <cases-dir>/labels.json")
    parser.add_argument("--cases", type=Path, help="default: <cases-dir>/cases.jsonl")
    parser.add_argument("--facts", type=Path, help="default: <cases-dir>/vlm/facts.jsonl")
    parser.add_argument("--frames", type=Path, help="default: <cases-dir>/frames")
    parser.add_argument("--review", type=Path, default=REVIEW, help="human reviewer notes per event")
    parser.add_argument("--out", type=Path, default=Path("data/casebase"))
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--eval", action="store_true", help="replay the bundle leave-one-setup-out and print metrics")
    parser.add_argument("--no-build", action="store_true", help="skip building (evaluate an existing bundle)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.no_build:
        header = build(args.labels or args.cases_dir / "labels.json", args.cases or args.cases_dir / "cases.jsonl",
                       args.facts or args.cases_dir / "vlm" / "facts.jsonl", args.frames or args.cases_dir / "frames",
                       args.review, args.out, args.max_side, args.quality)
        print(json.dumps({**header["counts"], "facts": header["facts"]}, ensure_ascii=False))
        print(f"bundle written to {args.out}")
    if args.eval:
        from cashwatch.casebase import CaseBase
        from cashwatch.casebase.evaluate import evaluate, format_report
        print(format_report(evaluate(CaseBase.load(args.out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
