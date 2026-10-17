"""Command line: uv run kb cases <stage> ..."""

import argparse
import json
import logging
import sys

from . import config


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "transformers", "qwen_vl_utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="kb")
    sub = parser.add_subparsers(dest="kb", required=True)
    cases = sub.add_parser("cases", help="empirical case memory from the labeled training videos")
    stages = cases.add_subparsers(dest="stage", required=True)
    stages.add_parser("labels", help="spreadsheet -> labels.json, probe videos")
    s = stages.add_parser("frames", help="decode frames, build windows, export event clips")
    s.add_argument("--force", action="store_true")
    s = stages.add_parser("annotate", help="VLM passes through Ollama (resumable)")
    s.add_argument("--passes", default="events,overview,facts")
    s.add_argument("--videos", default="", help="comma-separated video ids (default: all)")
    stages.add_parser("cases", help="assemble cases.jsonl and text cards")
    s = stages.add_parser("embed", help="Qwen3-VL-Embedding vectors")
    s.add_argument("--spaces", default="clip,frame,card")
    s.add_argument("--batch-size", type=int, default=8)
    s = stages.add_parser("eval", help="leave-one-setup-out evaluation")
    s.add_argument("--spaces", default="clip,frame,card")
    s.add_argument("--k", type=int, default=10)
    s = stages.add_parser("all", help="labels, frames, annotate, cases, embed, eval")
    s.add_argument("--skip-annotate", action="store_true")
    args = parser.parse_args(argv)

    from .cases import pipeline as pl
    p = pl.Paths(config.CASES_DIR)
    st = args.stage
    if st in ("labels", "all"):
        pl.stage_labels(p)
    if st in ("frames", "all"):
        pl.stage_frames(p, force=getattr(args, "force", False))
    if st == "annotate" or (st == "all" and not args.skip_annotate):
        passes = args.passes.split(",") if st == "annotate" else ["events", "overview", "facts"]
        videos = [v for v in getattr(args, "videos", "").split(",") if v]
        pl.stage_annotate(p, passes, videos or None)
    if st in ("cases", "all"):
        pl.stage_cases(p)
    if st in ("embed", "all"):
        spaces = args.spaces.split(",") if st == "embed" else ["clip", "frame", "card"]
        pl.stage_embed(p, spaces, getattr(args, "batch_size", 8))
    if st in ("eval", "all"):
        from .cases.evaluate import evaluate
        spaces = args.spaces.split(",") if st == "eval" else ["clip", "frame", "card"]
        res = evaluate(p.vectors, p.labels, spaces, getattr(args, "k", 10))
        from .cases.evaluate import load_event_types
        from .cases.facts_eval import evaluate_facts
        labels = {v["video"]: v for v in json.loads(p.labels.read_text(encoding="utf-8"))["videos"]}
        facts = evaluate_facts(p.vlm, labels, p.root / "flag_priors.json", load_event_types(p.cases))
        if facts:
            res["facts"] = facts
        (p.root / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
