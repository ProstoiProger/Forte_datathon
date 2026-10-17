"""Leave-one-setup-out evaluation of the case memory.

For every camera setup (group) the memory is rebuilt without that setup and the held-out windows are scored
against it, so no video is ever matched to itself or to a near-copy recorded in the same room. Two scorers:
  knn    share of event windows among the k nearest labeled windows (event = 1, normal = 0)
  probe  logistic regression on the embeddings, trained on event vs normal windows of the other setups
Reported per scorer: window AUC (event vs normal windows), video AUC (max window score, violation vs clean
videos), localization of the best window of each violation video, and type agreement of the nearest event case.
The zero-shot VLM baseline from Datathon/runs/vlm_video_eval has a video AUC of 0.54.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..store import Space

log = logging.getLogger(__name__)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def _knn_scores(space: Space, train: np.ndarray, test: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    sims = space.vectors[test] @ space.vectors[train].T
    k = min(k, len(train))
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    return y[train][top].mean(axis=1)


def _probe_scores(space: Space, train: np.ndarray, test: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    if len(set(y[train])) < 2:
        return np.zeros(len(test))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(space.vectors[train], y[train])
    return clf.predict_proba(space.vectors[test])[:, 1]


def loso_scores(space: Space, scorer: str, k: int = 10) -> np.ndarray:
    meta = space.meta
    groups = np.array([m["group"] for m in meta])
    rel = np.array([m["relation"] for m in meta])
    y = (rel == "event").astype(float)
    labeled = np.isin(rel, ["event", "normal"])
    scores = np.zeros(len(meta))
    for g in sorted(set(groups)):
        test = np.where(groups == g)[0]
        train = np.where((groups != g) & labeled)[0]
        if scorer == "knn":
            scores[test] = _knn_scores(space, train, test, y, k)
        else:
            scores[test] = _probe_scores(space, train, test, y)
    return scores


def _center(m: dict) -> float:
    return (m["start"] + m["end"]) / 2 if "start" in m else m["t"]


def report(space: Space, scores: np.ndarray, labels: dict) -> dict:
    meta = space.meta
    rel = [m["relation"] for m in meta]
    win_auc = auc([s for s, r in zip(scores, rel) if r == "event"], [s for s, r in zip(scores, rel) if r == "normal"])
    by_video: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by_video.setdefault(m["video"], []).append(i)
    vid_pos, vid_neg, errors = [], [], []
    for vid, idx in by_video.items():
        best = max(idx, key=lambda i: scores[i])
        if labels[vid]["label"] == "violation":
            vid_pos.append(scores[best])
            errors.append(min(abs(_center(meta[best]) - e["t"]) for e in labels[vid]["events"]))
        else:
            vid_neg.append(scores[best])
    errors = np.asarray(errors)
    return {
        "window_auc": round(win_auc, 3) if win_auc is not None else None,
        "video_auc": round(auc(vid_pos, vid_neg), 3),
        "violation_videos": len(vid_pos),
        "clean_videos": len(vid_neg),
        "best_window_err_median_s": round(float(np.median(errors)), 2),
        "best_window_hit_2s": int((errors <= 2).sum()),
        "best_window_hit_5s": int((errors <= 5).sum()),
    }


def type_agreement(space: Space, event_types: dict[str, str] | None = None) -> dict:
    """Nearest event window from another setup: does its type match?

    `event_types` (event id -> type, from cases.jsonl) carries manual review; window metadata holds the
    original hypothesis from labels.json and is the fallback.
    """
    def typ(m: dict) -> str | None:
        return (event_types or {}).get(m.get("event_id"), m.get("event_type"))

    ev = [i for i, m in enumerate(space.meta) if m["relation"] == "event" and typ(m)]
    hits, total = 0, 0
    for i in ev:
        cand = [j for j in ev if space.meta[j]["group"] != space.meta[i]["group"]]
        if not cand:
            continue
        j = cand[int(np.argmax(space.vectors[cand] @ space.vectors[i]))]
        hits += typ(space.meta[j]) == typ(space.meta[i])
        total += 1
    majority = Counter(typ(space.meta[i]) for i in ev).most_common(1)[0][1] / len(ev) if ev else 0
    return {"event_windows": total, "nn_type_match": round(hits / total, 3) if total else None,
            "majority_type_share": round(majority, 3)}


def center_per_video(space: Space) -> Space:
    """Subtract each video's median vector: removes the room's appearance, keeps what changes inside the video."""
    vids = np.array([m["video"] for m in space.meta])
    out = np.zeros_like(space.vectors)
    for v in set(vids):
        idx = np.where(vids == v)[0]
        out[idx] = space.vectors[idx] - np.median(space.vectors[idx], axis=0)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return Space(space.name + "_centered", (out / norm).astype(np.float32), space.meta)


def load_event_types(cases_path: Path) -> dict[str, str]:
    """Event id -> type from cases.jsonl (manual review applied); empty if the cases stage has not run."""
    event_types: dict[str, str] = {}
    if cases_path.exists():
        with cases_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    event_types.update({e["event_id"]: e["type"]["id"] for e in json.loads(line)["events"]})
    return event_types


def evaluate(vectors_dir: Path, labels_path: Path, spaces: list[str], k: int = 10) -> dict:
    labels = {v["video"]: v for v in json.loads(labels_path.read_text(encoding="utf-8"))["videos"]}
    event_types = load_event_types(labels_path.parent / "cases.jsonl")
    out = {}
    for name in spaces:
        if not (vectors_dir / f"{name}.npy").exists():
            continue
        space = Space.load(vectors_dir, name)
        if name == "card":  # only window cards carry a timeline position
            keep = [i for i, m in enumerate(space.meta) if m["kind"] == "window"]
            space = Space("card_window", space.vectors[keep], [space.meta[i] for i in keep])
        res = {"n": len(space), "relations": dict(Counter(m["relation"] for m in space.meta))}
        for scorer in ("knn", "probe"):
            res[scorer] = report(space, loso_scores(space, scorer, k), labels)
        if name == "clip":
            res["type"] = type_agreement(space, event_types)
            centered = center_per_video(space)
            res["centered"] = {s: report(centered, loso_scores(centered, s, k), labels) for s in ("knn", "probe")}
        out[name] = res
    return out
