import argparse
import json
import sys
from pathlib import Path

import cv2

from cashwatch.video import frames_at, probe

STATUS_RANK = {"confirmed": 0, "likely": 1, "uncertain": 2}


def choose_events(labels: dict, exclude_groups: set[str], statuses: set[str], per_type: int) -> dict[str, list[dict]]:
    chosen: dict[str, list[dict]] = {}
    candidates = []
    for v in labels["videos"]:
        if v["group"] in exclude_groups:
            continue
        for e in v["events"]:
            if e.get("type") and e.get("type_status", "unknown") in statuses:
                candidates.append((STATUS_RANK.get(e["type_status"], 9), v, e))
    for _, v, e in sorted(candidates, key=lambda c: c[0]):
        bucket = chosen.setdefault(e["type"], [])
        if len(bucket) < per_type and all(x["video"] != v["video"] for x in bucket):
            bucket.append({"video": v["video"], "file": v["file"], "group": v["group"], "t": float(e["t"]), "status": e["type_status"]})
    return chosen


def choose_normal(labels: dict, exclude_groups: set[str], count: int) -> list[dict]:
    clean = [v for v in labels["videos"] if v["label"] == "no_violation" and v["group"] not in exclude_groups]
    picked, groups = [], set()
    for v in clean:
        if v["group"] in groups:
            continue
        groups.add(v["group"])
        picked.append({"video": v["video"], "file": v["file"], "group": v["group"], "t": None, "status": "clean"})
        if len(picked) >= count:
            break
    return picked


def save_frame(video: Path, t: float | None, dest: Path, max_side: int) -> float:
    if t is None:
        t = probe(video).duration_sec / 2
    (actual, frame), = frames_at(video, [t])
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"cannot encode frame from {video}")
    dest.write_bytes(buf.tobytes())
    return round(actual, 2)


def build(labels: dict, videos: Path, out: Path, exclude_groups: set[str], statuses: set[str], per_type: int, normal: int, max_side: int) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    plan = choose_events(labels, exclude_groups, statuses, per_type)
    for type_id in sorted(plan):
        for i, item in enumerate(plan[type_id], 1):
            name = f"{type_id}__{i}.jpg"
            t = save_frame(videos / item["file"], item["t"], out / name, max_side)
            manifest.append({"type_id": type_id, "file": name, "source_video": item["video"], "group": item["group"], "t": t, "status": item["status"]})
    for i, item in enumerate(choose_normal(labels, exclude_groups, normal), 1):
        name = f"normal__{i}.jpg"
        t = save_frame(videos / item["file"], None, out / name, max_side)
        manifest.append({"type_id": "normal", "file": name, "source_video": item["video"], "group": item["group"], "t": t, "status": item["status"]})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract one reference frame per violation type for few-shot prompting")
    parser.add_argument("--labels", type=Path, default=Path("data/labels.json"))
    parser.add_argument("--videos", type=Path, default=Path("data/open/videos"))
    parser.add_argument("--out", type=Path, default=Path("data/references"))
    parser.add_argument("--exclude-groups", default="s11,s12", help="holdout groups never used as references")
    parser.add_argument("--status", default="confirmed,likely", help="label statuses allowed as references")
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument("--normal", type=int, default=1, help="number of frames of normal work from clean videos")
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args(argv)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    exclude = {g for g in args.exclude_groups.split(",") if g}
    manifest = build(labels, args.videos, args.out, exclude, set(args.status.split(",")), args.per_type, args.normal, args.max_side)
    for m in manifest:
        print(f"{m['file']:34s} {m['source_video']:6s} {m['group']} t={m['t']}s {m['status']}")
    print(f"{len(manifest)} reference frames written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
