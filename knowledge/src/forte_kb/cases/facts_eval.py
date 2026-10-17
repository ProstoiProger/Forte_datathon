"""Leave-one-setup-out evaluation of the VLM facts layer (per-window concept states).

Each 4 s facts window becomes a feature vector: whether each concept is on in any frame, whether it flips inside
the window, whether it differs from the video's usual state, and a few combinations that mirror violation types
(cashier leaving while cash lies on the desk, safe open while the cashier is away, ...). Two scorers:
  rules  no training: the number of rule-like flags that are on (transparent, unseen setups cannot leak)
  probe  logistic regression trained on event vs normal windows of the other setups
"""

import numpy as np

from .annotate import CONCEPTS, load_pass
from .evaluate import report
from .windows import relation
from ..store import Space

BOOL = [k for k in CONCEPTS if k != "people"]
RULES = {
    "leaving": lambda f: f["cashier_leaving"],
    "leaving_cash_on_desk": lambda f: f["cashier_leaving"] and f["cash_on_desk"],
    "leaving_safe_open": lambda f: f["cashier_leaving"] and f["safe_open"],
    "safe_open_unattended": lambda f: f["safe_open"] and not f["cashier_at_workplace"],
    "cash_unattended": lambda f: f["cash_on_desk"] and not f["cashier_at_workplace"],
    "phone": lambda f: f["phone_in_hand"],
    "drinking": lambda f: f["drinking"],
    "personal_items": lambda f: f["personal_items_on_desk"],
    "door_open": lambda f: f["door_open"],
    "crowd": lambda f: f["people"] >= 3,
}


def _frame_states(rec: dict) -> list[dict]:
    out = []
    for fr in rec["parsed"].get("frames", []):
        s = {k: bool(fr.get(k)) for k in BOOL}
        s["people"] = int(fr.get("people") or 0)
        out.append(s)
    return out


def window_features(records: list[dict]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Features for the windows of ONE video, in time order. Returns (X, names, rule_score)."""
    states = [_frame_states(r) for r in records]
    allf = [s for ws in states for s in ws]
    usual = {k: np.mean([s[k] for s in allf]) >= 0.5 for k in BOOL} if allf else {k: False for k in BOOL}
    usual_people = float(np.median([s["people"] for s in allf])) if allf else 0.0
    rows, rule_scores, prev = [], [], None
    for ws in states:
        ws = ws or [{**{k: False for k in BOOL}, "people": 0}]
        on = {k: any(s[k] for s in ws) for k in BOOL}
        flip = {k: len({s[k] for s in ws}) > 1 or (prev is not None and prev[k] != on[k]) for k in BOOL}
        unusual = {k: any(s[k] != usual[k] for s in ws) for k in BOOL}
        people = max(s["people"] for s in ws)
        rules = {name: any(fn(s) for s in ws) for name, fn in RULES.items()}
        row = ([on[k] for k in BOOL] + [flip[k] for k in BOOL] + [unusual[k] for k in BOOL]
               + [people, people - usual_people] + list(rules.values()))
        rows.append(row)
        rule_scores.append(sum(rules.values()))
        prev = on
    names = ([f"on:{k}" for k in BOOL] + [f"flip:{k}" for k in BOOL] + [f"unusual:{k}" for k in BOOL]
             + ["people", "people_vs_usual"] + [f"rule:{k}" for k in RULES])
    return np.asarray(rows, dtype=np.float32), names, np.asarray(rule_scores, dtype=np.float32)


def flag_priors(X: np.ndarray, names: list[str], meta: list[dict], labels: dict) -> dict:
    """How often each flag is on in event vs normal windows, and in how many violation vs clean videos.

    A flag that is common in clean videos (a phone in hand, the door opening) is normal work in this setting and
    must not raise a violation on its own; a flag never seen in clean videos is strong evidence.
    """
    rel = np.array([m["relation"] for m in meta])
    vids = np.array([m["video"] for m in meta])
    viol = {v for v, x in labels.items() if x["label"] == "violation"}
    n_viol, n_clean = len(viol), len(labels) - len(viol)
    out = {}
    for j, name in enumerate(names):
        if name in ("people", "people_vs_usual"):
            continue
        on = X[:, j] > 0
        if not on.any():
            continue
        flagged = set(vids[on])
        out[name] = {
            "event_window_rate": round(float(on[rel == "event"].mean()), 3),
            "normal_window_rate": round(float(on[rel == "normal"].mean()), 3),
            "violation_videos": f"{len(flagged & viol)}/{n_viol}",
            "clean_videos": f"{len(flagged - viol)}/{n_clean}",
        }
    return out


def evaluate_facts(vlm_dir, labels: dict, priors_path=None, event_types: dict | None = None) -> dict | None:
    facts = load_pass(vlm_dir, "facts")
    if not facts:
        return None
    by_video: dict[str, list[dict]] = {}
    for r in facts.values():
        by_video.setdefault(r["video"], []).append(r)
    X, meta, rule = [], [], []
    for vid, recs in by_video.items():
        recs.sort(key=lambda r: r["start"])
        x, names, rs = window_features(recs)
        v = labels[vid]
        for r, row, s in zip(recs, x, rs):
            rel, ev = relation(r["start"], r["end"], v["events"], v["label"] == "violation")
            meta.append({"id": r["key"], "video": vid, "group": v["group"], "start": r["start"], "end": r["end"],
                         "relation": rel, "event_id": ev["event_id"] if ev and rel == "event" else None,
                         "event_type": ev["type"] if ev and rel == "event" else None})
            X.append(row)
            rule.append(s)
    X = np.asarray(X)
    if priors_path is not None:
        import json
        priors_path.write_text(json.dumps(flag_priors(X, names, meta, labels), ensure_ascii=False, indent=1),
                               encoding="utf-8")
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    space = Space("facts", ((X - mu) / sd).astype(np.float32), meta)  # standardized features, not unit vectors
    from .evaluate import loso_scores, type_agreement
    norm = np.linalg.norm(space.vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1
    unit = Space("facts_unit", space.vectors / norm, meta)  # cosine nearest neighbour for the type check
    return {"n": len(space), "features": len(names),
            "rules": report(space, np.asarray(rule), labels),
            "probe": report(space, loso_scores(space, "probe"), labels),
            "type": type_agreement(unit, event_types)}
