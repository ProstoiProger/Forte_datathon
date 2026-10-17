"""Live progress bar for a VLM annotation pass, read from its log.

    uv run --no-sync python scripts/watch_progress.py [log] [--every 2]

Default log: store/cases/annotate_facts.log. Exits when the pass is complete; Ctrl+C to stop earlier.
"""

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

LINE = re.compile(r"^(\d\d:\d\d:\d\d) (\w+) (\d+)/(\d+) (\S+) ([\d.]+)s")


def snapshot(log: Path) -> dict | None:
    if not log.exists():
        return None
    rows = [m for m in (LINE.match(l) for l in log.read_text(encoding="utf-8", errors="replace").splitlines()) if m]
    if not rows:
        return None
    last = rows[-1]
    secs = [float(m.group(6)) for m in rows[-30:]]
    return {"pass": last.group(2), "done": int(last.group(3)), "total": int(last.group(4)), "key": last.group(5),
            "at": last.group(1), "avg": sum(secs) / len(secs),
            "fails": log.read_text(encoding="utf-8", errors="replace").count("PARSE FAIL")}


def render(s: dict, width: int = 40) -> str:
    frac = s["done"] / s["total"]
    filled = int(width * frac)
    left = (s["total"] - s["done"]) * s["avg"]
    eta = (dt.datetime.now() + dt.timedelta(seconds=left)).strftime("%H:%M")
    return (f"{s['pass']} [{'█' * filled}{'░' * (width - filled)}] {s['done']}/{s['total']} {frac:4.0%} | "
            f"{s['avg']:.1f}s/item | left ~{left / 60:.0f} min | ETA {eta} | last {s['key']} | fails {s['fails']}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="store/cases/annotate_facts.log", type=Path)
    parser.add_argument("--every", type=float, default=2.0, help="refresh period, seconds")
    args = parser.parse_args()
    try:
        while True:
            s = snapshot(args.log)
            line = render(s) if s else f"waiting for {args.log} ..."
            print("\r" + line.ljust(150), end="", flush=True)
            if s and s["done"] >= s["total"]:
                print("\ndone")
                return 0
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
