"""Offline leave-one-setup-out replay of the stored facts through CaseBase.candidates() (CPU only, no model calls).

Every training video is scored with exclude_groups={its setup}: rule weights and the threshold come from the other
setups only (the threshold itself is chosen by a nested leave-one-setup-out inside them). Reported: window AUC (event
vs normal windows), video AUC (best window score), flagged videos (TP/FN/FP/TN), localization of the best candidate
vs the labeled second, and type agreement for events with a registry type.
"""

from statistics import median
from typing import Any

from cashwatch.casebase.store import CaseBase

USEFUL_SEC = 5.0  # a candidate is useful when it is this close to a labeled second and has the event's type
MINI_SET = ("X-20", "X-21", "X-24", "X-25", "X-30", "X-31", "X-40", "X-43", "X-14")


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _center(start: float, end: float) -> float:
    return (start + end) / 2


def evaluate(cb: CaseBase, max_candidates: int = 2) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    win_pos: list[float] = []
    win_neg: list[float] = []
    for video in cb.videos():
        info = cb.video_info(video)
        exclude = frozenset({info["group"]})
        facts = cb.video_windows(video)
        duration = cb.video_duration(video)
        scores = cb.score_windows(facts, exclude, duration)
        for stored, ws in zip(cb.stored_windows(video), scores):
            if stored.relation == "event":
                win_pos.append(ws.score)
            elif stored.relation == "normal":
                win_neg.append(ws.score)
        cands = cb.candidates(facts, exclude, max_candidates, duration=duration)
        best = max(scores, key=lambda s: (s.score, -s.window.start))
        times = [e["t"] for e in info["events"]]

        def err(start: float, end: float) -> float | None:
            return min(abs(_center(start, end) - t) for t in times) if times else None

        useful = any(abs(_center(c.window.start, c.window.end) - e["t"]) <= USEFUL_SEC
                     and (e["type_id"] is None or c.type_id == e["type_id"]) for c in cands for e in info["events"])
        rows.append({
            **info,
            "useful": useful,
            "threshold": round(cb.model(exclude).threshold, 3),
            "best_score": round(best.score, 3),
            "best_window": [best.window.start, best.window.end, best.rule],
            "best_window_err": err(best.window.start, best.window.end),
            "flagged": bool(cands),
            "candidate_err": err(cands[0].window.start, cands[0].window.end) if cands else None,
            "candidates": [{"start": c.window.start, "end": c.window.end, "type_id": c.type_id, "rule": c.rule,
                            "score": c.score} for c in cands],
        })
    return _report(cb, rows, win_pos, win_neg)


def _report(cb: CaseBase, rows: list[dict], win_pos: list[float], win_neg: list[float]) -> dict[str, Any]:
    viol = [r for r in rows if r["label"] == "violation"]
    clean = [r for r in rows if r["label"] != "violation"]
    tp = sum(r["flagged"] for r in viol)
    fp = sum(r["flagged"] for r in clean)
    cand_err = [r["candidate_err"] for r in viol if r["candidate_err"] is not None]
    best_err = [r["best_window_err"] for r in viol if r["best_window_err"] is not None]
    typed = [(r, e) for r in viol for e in r["events"] if e["type_id"]]
    top_match = sum(1 for r, e in typed if r["candidates"] and r["candidates"][0]["type_id"] == e["type_id"])
    any_match = sum(1 for r, e in typed if any(c["type_id"] == e["type_id"] for c in r["candidates"]))
    mini = {r["video"]: r for r in rows if r["video"] in MINI_SET}
    return {
        "config": cb.config.describe(),
        "window_auc": _round(auc(win_pos, win_neg)),
        "video_auc": _round(auc([r["best_score"] for r in viol], [r["best_score"] for r in clean])),
        "videos": {"tp": tp, "fn": len(viol) - tp, "fp": fp, "tn": len(clean) - fp,
                   "useful": sum(r["useful"] for r in viol),
                   "recall": _round(tp / len(viol) if viol else None),
                   "false_alarm_rate": _round(fp / len(clean) if clean else None)},
        "localization": {
            "candidate_hit_2s": sum(e <= 2 for e in cand_err), "candidate_hit_5s": sum(e <= 5 for e in cand_err),
            "candidate_err_median_s": _round(median(cand_err) if cand_err else None),
            "best_window_hit_2s": sum(e <= 2 for e in best_err), "best_window_hit_5s": sum(e <= 5 for e in best_err),
            "best_window_err_median_s": _round(median(best_err) if best_err else None),
            "violation_videos": len(viol),
        },
        "type": {"registry_events": len(typed), "top_candidate_match": top_match, "any_candidate_match": any_match},
        "mini_set": {"violation_flagged": sum(r["flagged"] for r in mini.values() if r["label"] == "violation"),
                     "violation_videos": sum(r["label"] == "violation" for r in mini.values()),
                     "clean_flagged": sorted(v for v, r in mini.items() if r["label"] != "violation" and r["flagged"]),
                     "clean_videos": sum(r["label"] != "violation" for r in mini.values())},
        "rows": rows,
    }


def _round(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def format_report(rep: dict[str, Any], rows: bool = True) -> str:
    v, loc, ty, mini = rep["videos"], rep["localization"], rep["type"], rep["mini_set"]
    lines = [
        f"config: {rep['config']['combine']}, threshold={rep['config']['threshold'] or 'nested LOSO'}",
        f"window AUC {rep['window_auc']}  video AUC {rep['video_auc']}",
        f"videos: TP {v['tp']} FN {v['fn']} FP {v['fp']} TN {v['tn']}  (recall {v['recall']}, "
        f"false alarms {v['false_alarm_rate']}); useful (right place and type) {v['useful']}/{v['tp'] + v['fn']}",
        f"best candidate within 2 s: {loc['candidate_hit_2s']}/{loc['violation_videos']}, within 5 s: "
        f"{loc['candidate_hit_5s']}/{loc['violation_videos']} (median err {loc['candidate_err_median_s']} s); "
        f"best window regardless of threshold: {loc['best_window_hit_2s']} / {loc['best_window_hit_5s']}",
        f"type: top candidate {ty['top_candidate_match']}/{ty['registry_events']}, any candidate "
        f"{ty['any_candidate_match']}/{ty['registry_events']}",
        f"mini set: violations {mini['violation_flagged']}/{mini['violation_videos']}, clean flagged "
        f"{len(mini['clean_flagged'])}/{mini['clean_videos']} {mini['clean_flagged']}",
    ]
    if rows:
        for r in rep["rows"]:
            events = ",".join(f"{e['t']:g}:{e['type_id'] or 'new'}" for e in r["events"]) or "-"
            cands = "; ".join(f"{c['start']:g}-{c['end']:g} {c['type_id'] or '?'}/{c['rule']} {c['score']:.2f}"
                              for c in r["candidates"]) or "-"
            if r["label"] == "violation":
                mark = ("TP*" if r["useful"] else "TP ") if r["flagged"] else "FN "
            else:
                mark = "FP " if r["flagged"] else "TN "
            lines.append(f"  {mark} {r['video']:5s} {r['group']} th={r['threshold']:.2f} best={r['best_score']:.2f} "
                         f"events[{events}] -> {cands}")
    return "\n".join(lines)
