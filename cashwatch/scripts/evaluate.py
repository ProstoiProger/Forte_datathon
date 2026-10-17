import argparse
import csv
import json
import sys
import time
from pathlib import Path

from cashwatch.analyzers import build_analyzer
from cashwatch.config import Settings
from cashwatch.evaluation import (
    DEFAULT_TAUS,
    build_rows,
    format_summary,
    holdout,
    logo_cv,
    pick_tau,
    score,
    sweep,
    video_report,
)
from cashwatch.transforms import TRANSFORMS, transform_video

SETTING_FLAGS = {
    "analyzer": str,
    "stub_mode": str,
    "vlm_url": str,
    "vlm_model": str,
    "vlm_fps": float,
    "vlm_window_sec": float,
    "vlm_stride_sec": float,
    "vlm_max_frames": int,
    "vlm_max_side": int,
    "vlm_num_ctx": int,
    "vlm_merge_gap_sec": float,
    "vlm_timeout_sec": float,
    "vlm_retries": int,
    "vlm_jpeg_quality": int,
    "vlm_max_windows": int,
    "vlm_score": str,
    "vlm_reference_dir": str,
    "vlm_max_references": int,
    "time_budget_sec": float,
    "fallback": str,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_videos(labels: dict, groups: str | None, exclude: str | None, only: str | None, limit: int | None) -> list[dict]:
    vids = labels["videos"]
    if groups:
        keep = set(groups.split(","))
        vids = [v for v in vids if v["group"] in keep]
    if exclude:
        drop = set(exclude.split(","))
        vids = [v for v in vids if v["group"] not in drop]
    if only:
        names = set(only.split(","))
        vids = [v for v in vids if v["video"] in names]
    return vids[:limit] if limit else vids


def cmd_run(args: argparse.Namespace) -> int:
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    overrides = {k: getattr(args, k) for k in SETTING_FLAGS if getattr(args, k) is not None}
    settings = Settings(_env_file=args.env_file, api_keys="evaluation", **overrides)
    analyzer = build_analyzer(settings)
    run_dir = args.runs_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in settings.model_dump(mode="json").items() if k != "api_keys"}
    config["transform"] = args.transform
    config["reference_exclusion"] = "same setup group as the scored video"
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    check = getattr(analyzer, "check", None)
    if check:
        ok, detail = check()
        print(f"analyzer {analyzer.name}: {'ready' if ok else 'NOT READY'} ({detail})")
        if not ok and not args.ignore_check:
            return 2
    pred_path = run_dir / "predictions.jsonl"
    done = {p["video"] for p in load_jsonl(pred_path)} if not args.restart else set()
    if args.restart:
        pred_path.unlink(missing_ok=True)
    todo = [v for v in select_videos(labels, args.groups, args.exclude_groups, args.videos_only, args.limit) if v["video"] not in done]
    print(f"run {args.name}: {len(todo)} videos to process, {len(done)} cached{' transform=' + args.transform if args.transform else ''}")
    inner = getattr(analyzer, "primary", analyzer)
    with pred_path.open("a", encoding="utf-8") as out:
        for i, v in enumerate(todo, 1):
            path = args.videos / v["file"]
            if args.transform:
                path = transform_video(path, args.transform_cache / args.transform / f"{v['video']}.mov", args.transform)
            if hasattr(inner, "exclude_reference_groups"):
                inner.exclude_reference_groups = {v["group"]}
            started = time.monotonic()
            record = {"video": v["video"], "group": v["group"], "events": [], "error": None}
            try:
                events = analyzer.analyze(path, lambda p: None, time.monotonic() + settings.time_budget_sec)
                record["events"] = [
                    {
                        "t": e.timecode_sec,
                        "type": e.type_id,
                        "conf": e.confidence,
                        "raw_conf": e.raw_confidence,
                        "votes": e.votes,
                        "coverage": e.coverage,
                        "observation": e.observation,
                    }
                    for e in events
                ]
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] = round(time.monotonic() - started, 2)
            stats = getattr(inner, "last_stats", None)
            if stats:
                record["stats"] = stats
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            top = max((e["conf"] for e in record["events"]), default=0.0)
            gt = ", ".join(f"{e['t']:g}s {e.get('type') or '?'}" for e in v["events"]) or "clean"
            print(f"[{i}/{len(todo)}] {v['video']:6s} {record['seconds']:6.1f}s events={len(record['events'])} max_conf={top:.2f} gt=({gt}){' ERROR ' + record['error'] if record['error'] else ''}")
    args.run = run_dir
    return cmd_score(args)


def cmd_score(args: argparse.Namespace) -> int:
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    predictions = load_jsonl(args.run / "predictions.jsonl")
    rows = build_rows(labels, predictions)
    if not rows:
        print("no predictions to score")
        return 1
    kw = {"tol": args.tol, "match_window": args.match_window}
    tau = args.tau if args.tau is not None else pick_tau(rows, DEFAULT_TAUS, args.min_recall, **kw)
    main = score(rows, tau, **kw)
    cv = logo_cv(rows, DEFAULT_TAUS, args.min_recall, **kw)
    result = {"tau": tau, "min_recall": args.min_recall, "all": main, "logo_cv": cv}
    lines = [
        format_summary("all videos, tau tuned on all (optimistic)" if args.tau is None else "all videos, fixed tau", main),
        format_summary("leave-one-setup-out (honest)", cv),
    ]
    if args.holdout:
        groups = set(args.holdout.split(","))
        ho = holdout(rows, groups, DEFAULT_TAUS, args.min_recall, **kw)
        result["holdout"] = ho
        lines.append(format_summary(f"holdout {','.join(sorted(groups))}, tau tuned on the rest", ho))
    table = sweep(rows, DEFAULT_TAUS, **kw)
    result["sweep"] = [
        {"tau": s["tau"], "accuracy": s["detection"]["accuracy"], "recall": s["detection"]["recall"], "fpr": s["detection"]["fpr"], "proxy_55": s["proxy_55"]}
        for s in table
    ]
    report = video_report(rows, tau, args.match_window)
    (args.run / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    with (args.run / "videos.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    print("\n".join(lines))
    print("\ntau   acc    recall fpr    proxy")
    for s in result["sweep"]:
        print(f"{s['tau']:.2f}  {s['accuracy']:.3f}  {s['recall']:.3f}  {s['fpr']:.3f}  {s['proxy_55']:.1f}")
    bad = [r for r in report if r["issues"]]
    print(f"\nvideos with issues at tau={tau:.2f}: {len(bad)}/{len(report)}")
    for r in bad:
        print(f"  {r['video']:6s} {r['group']} gt={r['gt']:9s} pred={r['pred']:9s} {r['issues']}")
    print(f"\nwritten {args.run / 'metrics.json'} and {args.run / 'videos.csv'}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an analyzer over the labeled videos and score it")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--labels", type=Path, default=Path("data/labels.json"))
        s.add_argument("--tau", type=float, default=None, help="fixed confidence threshold; tuned when omitted")
        s.add_argument("--min-recall", type=float, default=0.95)
        s.add_argument("--tol", type=float, default=2.0, help="timecode tolerance in seconds")
        s.add_argument("--match-window", type=float, default=5.0)
        s.add_argument("--holdout", default=None, help="comma-separated setup groups kept out of threshold tuning")
        if name == "score":
            s.add_argument("--run", type=Path, required=True)
        else:
            s.add_argument("--name", required=True)
            s.add_argument("--videos", type=Path, default=Path("data/open/videos"))
            s.add_argument("--runs-dir", type=Path, default=Path("runs"))
            s.add_argument("--env-file", default=None)
            s.add_argument("--groups", default=None)
            s.add_argument("--exclude-groups", default=None)
            s.add_argument("--videos-only", default=None, help="comma-separated video ids")
            s.add_argument("--limit", type=int, default=None)
            s.add_argument("--restart", action="store_true", help="drop cached predictions")
            s.add_argument("--ignore-check", action="store_true", help="run even if the model is not reachable")
            s.add_argument("--transform", choices=sorted(TRANSFORMS), default=None, help="evaluate on transformed copies of the videos")
            s.add_argument("--transform-cache", type=Path, default=Path("data/transformed"))
            for flag, kind in SETTING_FLAGS.items():
                s.add_argument(f"--{flag.replace('_', '-')}", dest=flag, type=kind, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return cmd_run(args) if args.command == "run" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
