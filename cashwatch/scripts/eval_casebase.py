"""Leave-one-setup-out replay of the stored facts through the case base: final configuration vs alternatives.

CPU only, no model calls. Needs a bundle built by scripts/build_casebase.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from cashwatch.casebase import CaseBase, ScoringConfig
from cashwatch.casebase.evaluate import evaluate, format_report

ALTERNATIVES: dict[str, ScoringConfig] = {
    "final": ScoringConfig(),
    "any_event_llr": ScoringConfig(typed_hits=False),  # rule hits counted in event windows of any type
    "sum_of_rules": ScoringConfig(combine="sum"),
    "rule_count": ScoringConfig(combine="count"),  # unweighted number of fired rules (closest to the baseline)
    "fixed_threshold_1": ScoringConfig(threshold=1.0),  # a priori: rule fires e times more often in violations
    "youden_threshold": ScoringConfig(recall_cost=1.0),  # misses and false alarms weighted equally
    "no_unusual_bonus": ScoringConfig(bonus_cap=0.0),
}


def summary_line(name: str, rep: dict) -> str:
    v, loc, ty, mini = rep["videos"], rep["localization"], rep["type"], rep["mini_set"]
    return (f"{name:18s} wAUC {rep['window_auc']:.3f} vAUC {rep['video_auc']:.3f} | TP {v['tp']:2d} FN {v['fn']:2d} "
            f"FP {v['fp']:2d} TN {v['tn']:2d} useful {v['useful']:2d} | loc 2s {loc['candidate_hit_2s']:2d} "
            f"5s {loc['candidate_hit_5s']:2d} | type top {ty['top_candidate_match']}/{ty['registry_events']} "
            f"any {ty['any_candidate_match']}/{ty['registry_events']} | mini {mini['violation_flagged']}/"
            f"{mini['violation_videos']} clean {len(mini['clean_flagged'])}/{mini['clean_videos']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the case base leave-one-setup-out on the stored facts")
    parser.add_argument("--bundle", type=Path, default=Path("data/casebase"))
    parser.add_argument("--only", help="comma-separated configuration names (default: all)")
    parser.add_argument("--rows", action="store_true", help="print per-video rows of the final configuration")
    parser.add_argument("--json", type=Path, help="write all reports to this file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    base = CaseBase.load(args.bundle)
    names = args.only.split(",") if args.only else list(ALTERNATIVES)
    reports = {}
    for name in names:
        reports[name] = evaluate(base.configured(ALTERNATIVES[name]))
        print(summary_line(name, reports[name]))
    if args.rows and "final" in reports:
        print(format_report(reports["final"]))
    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
