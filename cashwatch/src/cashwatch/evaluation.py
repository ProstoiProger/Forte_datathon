import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TAUS = tuple(round(i * 0.05, 2) for i in range(21))
WEIGHTS = {"detection": 25.0, "localization": 15.0, "classification": 15.0}


@dataclass(frozen=True)
class Pred:
    t: float
    type: str
    conf: float


@dataclass
class Row:
    video: str
    group: str
    violation: bool
    gt: list[tuple[float, str | None]]
    preds: list[Pred] = field(default_factory=list)
    seconds: float | None = None
    error: str | None = None


def build_rows(labels: dict, predictions: list[dict]) -> list[Row]:
    by_video = {p["video"]: p for p in predictions}
    rows = []
    for v in labels["videos"]:
        p = by_video.get(v["video"])
        if p is None:
            continue
        rows.append(
            Row(
                video=v["video"],
                group=v["group"],
                violation=v["label"] == "violation",
                gt=[(float(e["t"]), e.get("type")) for e in v["events"]],
                preds=[Pred(float(x["t"]), x["type"], float(x["conf"])) for x in p.get("events", [])],
                seconds=p.get("seconds"),
                error=p.get("error"),
            )
        )
    return rows


def kept(row: Row, tau: float) -> list[Pred]:
    return [p for p in row.preds if p.conf >= tau]


def match(gt: list[tuple[float, str | None]], preds: list[Pred], window: float) -> list[tuple[int, Pred | None]]:
    pairs = sorted(
        ((abs(g[0] - p.t), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(preds)),
        key=lambda x: x[0],
    )
    used_g, used_p, result = set(), set(), {}
    for dist, gi, pi in pairs:
        if dist > window or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        result[gi] = preds[pi]
    return [(gi, result.get(gi)) for gi in range(len(gt))]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def score(rows: list[Row], tau: float | dict[str, float], tol: float = 2.0, match_window: float = 5.0) -> dict[str, Any]:
    tp = fn = fp = tn = 0
    n_events = matched = within = within1 = typed = type_ok = type_ok_matched = 0
    errors = []
    for row in rows:
        t = tau[row.video] if isinstance(tau, dict) else tau
        preds = kept(row, t)
        flagged = bool(preds)
        if row.violation:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
        if not row.violation:
            continue
        for gi, pred in match(row.gt, preds, match_window):
            gt_t, gt_type = row.gt[gi]
            n_events += 1
            if gt_type is not None:
                typed += 1
            if pred is None:
                continue
            matched += 1
            dt = abs(pred.t - gt_t)
            errors.append(dt)
            within += dt <= tol
            within1 += dt <= 1.0
            if gt_type is not None and pred.type == gt_type:
                type_ok += 1
                type_ok_matched += 1
    n = tp + fn + fp + tn
    det = {
        "videos": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }
    loc = {
        "events": n_events,
        "matched": matched,
        "mae_sec": sum(errors) / len(errors) if errors else None,
        "within_tol": within / n_events if n_events else 0.0,
        "within_1s": within1 / n_events if n_events else 0.0,
        "tol_sec": tol,
    }
    cls = {
        "typed_events": typed,
        "accuracy": type_ok / typed if typed else 0.0,
        "accuracy_matched": type_ok_matched / matched if matched else 0.0,
    }
    seconds = [r.seconds for r in rows if r.seconds is not None]
    proxy = WEIGHTS["detection"] * det["accuracy"] + WEIGHTS["localization"] * loc["within_tol"] + WEIGHTS["classification"] * cls["accuracy"]
    return {
        "tau": tau if not isinstance(tau, dict) else None,
        "detection": det,
        "localization": loc,
        "classification": cls,
        "proxy_55": round(proxy, 2),
        "latency": {"p50": _percentile(seconds, 0.5), "p95": _percentile(seconds, 0.95), "max": max(seconds) if seconds else None},
        "errors": sum(r.error is not None for r in rows),
    }


def sweep(rows: list[Row], taus: tuple[float, ...] = DEFAULT_TAUS, **kw) -> list[dict[str, Any]]:
    return [score(rows, t, **kw) for t in taus]


def pick_tau(rows: list[Row], taus: tuple[float, ...] = DEFAULT_TAUS, min_recall: float = 0.95, **kw) -> float:
    results = sweep(rows, taus, **kw)
    ok = [r for r in results if r["detection"]["recall"] >= min_recall]
    pool = ok or results
    key = (lambda r: (r["proxy_55"], r["tau"])) if ok else (lambda r: (r["detection"]["recall"], r["proxy_55"], r["tau"]))
    return max(pool, key=key)["tau"]


def logo_cv(rows: list[Row], taus: tuple[float, ...] = DEFAULT_TAUS, min_recall: float = 0.95, **kw) -> dict[str, Any]:
    groups = sorted({r.group for r in rows})
    chosen: dict[str, float] = {}
    per_video: dict[str, float] = {}
    for g in groups:
        train = [r for r in rows if r.group != g]
        tau = pick_tau(train, taus, min_recall, **kw) if train else 0.5
        chosen[g] = tau
        for r in rows:
            if r.group == g:
                per_video[r.video] = tau
    result = score(rows, per_video, **kw)
    result["tau_by_group"] = chosen
    return result


def holdout(rows: list[Row], holdout_groups: set[str], taus: tuple[float, ...] = DEFAULT_TAUS, min_recall: float = 0.95, **kw) -> dict[str, Any]:
    train = [r for r in rows if r.group not in holdout_groups]
    test = [r for r in rows if r.group in holdout_groups]
    tau = pick_tau(train, taus, min_recall, **kw)
    result = score(test, tau, **kw)
    result["train_videos"] = len(train)
    return result


def video_report(rows: list[Row], tau: float, match_window: float = 5.0) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        preds = kept(r, tau)
        issues = []
        flagged = bool(preds)
        if r.violation and not flagged:
            issues.append("missed")
        if not r.violation and flagged:
            issues.append("false_positive")
        if r.violation and flagged:
            for gi, p in match(r.gt, preds, match_window):
                gt_t, gt_type = r.gt[gi]
                if p is None:
                    issues.append(f"event@{gt_t:g}s unmatched")
                else:
                    if abs(p.t - gt_t) > 2.0:
                        issues.append(f"event@{gt_t:g}s off by {p.t - gt_t:+.1f}s")
                    if gt_type and p.type != gt_type:
                        issues.append(f"event@{gt_t:g}s type {p.type} != {gt_type}")
        if r.error:
            issues.append(f"error: {r.error}")
        out.append(
            {
                "video": r.video,
                "group": r.group,
                "gt": "violation" if r.violation else "clean",
                "pred": "violation" if flagged else "clean",
                "max_conf": max((p.conf for p in r.preds), default=0.0),
                "gt_events": "; ".join(f"{t:g}s {ty or '?'}" for t, ty in r.gt),
                "pred_events": "; ".join(f"{p.t:.1f}s {p.type} {p.conf:.2f}" for p in sorted(preds, key=lambda p: p.t)),
                "seconds": r.seconds,
                "issues": "; ".join(issues),
            }
        )
    return out


def format_summary(title: str, s: dict[str, Any]) -> str:
    d, l, c, lat = s["detection"], s["localization"], s["classification"], s["latency"]
    mae = "-" if l["mae_sec"] is None else f"{l['mae_sec']:.2f}s"
    p95 = "-" if lat["p95"] is None else f"{lat['p95']:.1f}s"
    tau = "per-group" if s.get("tau") is None else f"{s['tau']:.2f}"
    return (
        f"{title} (tau={tau}, n={d['videos']}): "
        f"detection acc={d['accuracy']:.3f} recall={d['recall']:.3f} fpr={d['fpr']:.3f} [tp={d['tp']} fn={d['fn']} fp={d['fp']} tn={d['tn']}] | "
        f"localization within{l['tol_sec']:g}s={l['within_tol']:.3f} within1s={l['within_1s']:.3f} mae={mae} matched={l['matched']}/{l['events']} | "
        f"type acc={c['accuracy']:.3f} (matched {c['accuracy_matched']:.3f}) | proxy={s['proxy_55']:.1f}/55 | p95={p95} errors={s['errors']}"
    )
