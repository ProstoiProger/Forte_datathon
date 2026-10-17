import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EVALUATE = Path(__file__).resolve().parent / "evaluate.py"


def log(path: Path, message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_experiment(exp: dict, defaults: list[str], runs_dir: Path, force: bool, queue_log: Path) -> dict:
    name = exp["name"]
    run_dir = runs_dir / name
    if (run_dir / "metrics.json").exists() and not force:
        log(queue_log, f"skip {name}: already scored")
        return {"name": name, "status": "cached", "minutes": 0.0}
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(EVALUATE), "run", "--name", name, "--runs-dir", str(runs_dir), *defaults, *exp.get("args", [])]
    if force:
        cmd.append("--restart")
    env = {**os.environ, "PYTHONUTF8": "1"}
    timeout = exp.get("timeout_min")
    log(queue_log, f"start {name}: {' '.join(cmd[2:])}")
    started = time.monotonic()
    with (run_dir / "queue.log").open("a", encoding="utf-8") as fh:
        try:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout * 60 if timeout else None)
            status = "ok" if proc.returncode == 0 else f"failed rc={proc.returncode}"
        except subprocess.TimeoutExpired:
            status = f"timeout after {timeout} min"
    minutes = round((time.monotonic() - started) / 60, 1)
    log(queue_log, f"end {name}: {status} in {minutes} min")
    return {"name": name, "status": status, "minutes": minutes}


def leaderboard(runs_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cv, full = m["logo_cv"], m["all"]
        rows.append({
            "run": run_dir.name,
            "honest": cv["proxy_55"],
            "optimistic": full["proxy_55"],
            "tau": m["tau"],
            "recall": cv["detection"]["recall"],
            "fpr": cv["detection"]["fpr"],
            "within2s": cv["localization"]["within_tol"],
            "type_acc": cv["classification"]["accuracy"],
            "p95_sec": full["latency"]["p95"],
            "videos": full["detection"]["videos"],
            "errors": full["errors"],
            "model": cfg.get("vlm_model") if cfg.get("analyzer") == "vlm" else cfg.get("analyzer"),
            "transform": cfg.get("transform") or "-",
            "references": "yes" if cfg.get("vlm_reference_dir") else "no",
        })
    return sorted(rows, key=lambda r: (-r["honest"], r["run"]))


def render(rows: list[dict]) -> str:
    head = "| run | honest /55 | optimistic /55 | tau | recall | fpr | within 2s | type acc | p95 s | videos | errors | model | refs | transform |"
    lines = [head, "|" + "---|" * 14]
    for r in rows:
        p95 = "-" if r["p95_sec"] is None else f"{r['p95_sec']:.1f}"
        lines.append(
            f"| {r['run']} | {r['honest']:.1f} | {r['optimistic']:.1f} | {r['tau']:.2f} | {r['recall']:.2f} | {r['fpr']:.2f} | "
            f"{r['within2s']:.2f} | {r['type_acc']:.2f} | {p95} | {r['videos']} | {r['errors']} | {r['model']} | {r['references']} | {r['transform']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a queue of evaluation experiments unattended and build a leaderboard")
    parser.add_argument("experiments", type=Path, nargs="?", help="JSON file with defaults and experiments")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--only", default=None, help="comma-separated experiment names")
    parser.add_argument("--force", action="store_true", help="rerun experiments that already have metrics")
    parser.add_argument("--leaderboard-only", action="store_true")
    args = parser.parse_args(argv)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    queue_log = args.runs_dir / "queue.log"
    results = []
    if not args.leaderboard_only:
        if args.experiments is None:
            parser.error("experiments file is required unless --leaderboard-only is set")
        spec = json.loads(args.experiments.read_text(encoding="utf-8"))
        names = set(args.only.split(",")) if args.only else None
        experiments = [e for e in spec["experiments"] if names is None or e["name"] in names]
        log(queue_log, f"queue start: {len(experiments)} experiments")
        for exp in experiments:
            results.append(run_experiment(exp, spec.get("defaults", []), args.runs_dir, args.force, queue_log))
        log(queue_log, "queue done: " + ", ".join(f"{r['name']}={r['status']}" for r in results))
    table = render(leaderboard(args.runs_dir))
    (args.runs_dir / "leaderboard.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0 if all(r["status"] in ("ok", "cached") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
