"""Empirical case memory: labeled windows and events of the training videos, used to rank suspicious windows of a new
video (candidates) and to pick reference examples for the verifier.

Everything learned from the bundle (rule weights, the decision threshold, references) is computed only from camera
setups (groups) not in `exclude_groups`, so one object serves leave-one-setup-out evaluation (exclude the scored
video's setup) and production (exclude nothing). Learned models are cached per exclude set.

Scoring: each rule's weight is the smoothed log likelihood ratio of firing in a labeled violation window (of the rule's
type) vs a normal window. With half a pseudo-count per cell, a rule never seen in labeled windows keeps a prior weight
of log((N+1)/(E+1)) ~ 2: it was designed for a violation and is absent from normal work. A window scores its best rule
plus a bounded bonus for facts unusual for this video; without a typed rule, strongly unusual facts make an open-world
candidate (type None). The decision threshold is chosen leave-one-setup-out inside the kept setups.
"""

import base64
import json
import logging
import math
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cashwatch.analyzers.facts.schema import FLAG_KEYS, CaseRef, Candidate, FactWindow
from cashwatch.casebase.rules import GAP_TOLERANCE, RULES, UNUSUAL_FEATURES, UNUSUAL_PREFIX, Rule, fire

log = logging.getLogger("cashwatch.casebase")

BUNDLE_FILE = "casebase.json"
UNUSUAL_RULE = "unusual"
EVENT, NEAR, NORMAL = "event", "near", "normal"
COMBINE_MODES = ("max", "sum", "count")
SAME_RULE_BONUS = 1.0  # a normal reference where the candidate's rule also fires counts as one flag closer
WIDE_WINDOWS = 1.5  # runtime windows this many times longer than the case base windows are logged as a caveat


@dataclass(frozen=True)
class StoredWindow:
    id: str
    video: str
    group: str
    label: str
    start: float
    end: float
    relation: str  # event | near | other | normal
    event_id: str | None
    type_id: str | None
    flags: dict[str, bool]
    people: int
    scene_ru: str = ""
    image: str | None = None
    frame_times: tuple[float, ...] = ()


@dataclass(frozen=True)
class StoredEvent:
    event_id: str
    video: str
    group: str
    t: float
    type_id: str | None  # None for confirmed violations outside the registry ("new")
    title: str
    description_ru: str
    image: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)
    people: int = 0


@dataclass(frozen=True)
class ScoringConfig:
    combine: str = "max"  # max: best rule weight; sum: sum of positive rule weights; count: number of fired rules
    threshold: float | None = None  # fixed decision threshold; None = chosen leave-one-setup-out on the kept setups
    typed_hits: bool = True  # a rule hit in an event window counts only if the event has the rule's type
    alpha: float = 0.5  # pseudo-count per cell of the rule rates (Jeffreys prior)
    bonus_cap: float = 0.5  # facts unusual for this video add at most this much to a window with a typed rule
    open_world_min: float = 3.0  # without a typed rule, unusual facts make a candidate only if their LLR sum reaches
    # this (likelihood ratio ~20); the window then scores that sum
    recall_cost: float = 0.5  # threshold maximizes TPR - recall_cost * FPR over videos (a miss costs 2 false alarms)
    threshold_target: str = "useful"  # useful: a violation video counts as found only by a candidate of the right
    # type near a labeled second (what the verifier can confirm); any: by any candidate
    rules: tuple[Rule, ...] = RULES

    def __post_init__(self) -> None:
        if self.combine not in COMBINE_MODES:
            raise ValueError(f"combine must be one of {COMBINE_MODES}, got {self.combine!r}")
        if self.threshold_target not in ("useful", "any"):
            raise ValueError(f"threshold_target must be 'useful' or 'any', got {self.threshold_target!r}")

    def describe(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k != "rules"}
        out["rules"] = [r.name for r in self.rules]
        return out


@dataclass(frozen=True)
class Model:
    """Statistics learned from the kept setups."""

    groups: frozenset[str]
    weights: dict[str, float]  # rule or unusual feature -> smoothed log likelihood ratio (event vs normal windows)
    hits: dict[str, tuple[int, int]]  # name -> (event windows, normal windows) where it fires
    event_windows: int
    normal_windows: int
    threshold: float


@dataclass(frozen=True)
class WindowScore:
    window: FactWindow
    score: float
    rule: str | None  # best typed rule; "unusual" for an open-world window; None when nothing counts
    type_id: str | None
    typed: tuple[tuple[str, str | None, float], ...]  # (rule, type_id, weight) of the counted rules, best first
    bonus: float  # capped unusual bonus included in score
    unusual: float  # sum of the positive weights of the unusual facts (uncapped)
    fired: frozenset[str] = frozenset()

    def pairs(self, combine: str = "max") -> list[tuple[float, str, str | None]]:
        """(score, rule, type_id) per rule type counted here; the best type gets the window score."""
        if not self.typed:
            return [(self.score, UNUSUAL_RULE, None)] if self.rule == UNUSUAL_RULE else []
        out, seen = [], set()
        for rank, (rule, type_id, weight) in enumerate(self.typed):
            if type_id in seen:
                continue
            seen.add(type_id)
            gap = rank if combine == "count" else self.typed[0][2] - weight
            out.append((self.score - gap, rule, type_id))
        return out


def log_likelihood_ratio(hit_event: int, n_event: int, hit_normal: int, n_normal: int, alpha: float) -> float:
    p_event = (hit_event + alpha) / (n_event + 2 * alpha)
    p_normal = (hit_normal + alpha) / (n_normal + 2 * alpha)
    return math.log(p_event / p_normal)


def choose_threshold(video_scores: Sequence[tuple[bool, float]], recall_cost: float) -> float | None:
    """Threshold on the best window score of a video maximizing TPR - recall_cost * FPR; ties keep the lower one.

    Returns the midpoint between the chosen score and the next lower one (margin towards recall), None if no video
    has a positive score.
    """
    values = sorted({s for _, s in video_scores if s > 0})
    if not values:
        return None
    n_pos = sum(1 for v, _ in video_scores if v) or 1
    n_neg = sum(1 for v, _ in video_scores if not v) or 1
    best, best_gain = values[0], -math.inf
    for th in values:
        tpr = sum(1 for v, s in video_scores if v and s >= th) / n_pos
        fpr = sum(1 for v, s in video_scores if not v and s >= th) / n_neg
        gain = tpr - recall_cost * fpr
        if gain > best_gain + 1e-12:
            best, best_gain = th, gain
    lower = max((s for s in values if s < best), default=0.0)
    return (best + lower) / 2


def flag_distance(a_flags: dict[str, bool], a_people: int, b_flags: dict[str, bool], b_people: int) -> float:
    differ = sum(bool(a_flags.get(k)) != bool(b_flags.get(k)) for k in FLAG_KEYS)
    return differ + 0.5 * min(abs(a_people - b_people), 2)


def _touch(a: FactWindow, b: FactWindow) -> bool:
    return b.start - a.end <= GAP_TOLERANCE and a.start - b.end <= GAP_TOLERANCE


def to_fact_window(w: StoredWindow, index: int) -> FactWindow:
    return FactWindow(index=index, start=w.start, end=w.end, frame_times=w.frame_times,
                      flags={k: bool(w.flags.get(k)) for k in FLAG_KEYS}, people=w.people)


class CaseBase:
    def __init__(self, directory: Path, windows: Sequence[StoredWindow], events: Sequence[StoredEvent],
                 header: dict | None = None, config: ScoringConfig | None = None,
                 durations: dict[str, float] | None = None) -> None:
        self.directory = Path(directory)
        self.header = header or {}
        self.config = config or ScoringConfig()
        self._windows = list(windows)
        self._events = list(events)
        self._by_video: dict[str, list[StoredWindow]] = {}
        for w in self._windows:
            self._by_video.setdefault(w.video, []).append(w)
        for ws in self._by_video.values():
            ws.sort(key=lambda w: w.start)
        self._durations = dict(durations or {})
        self._fired: dict[str, frozenset[str]] = {}
        for video, ws in self._by_video.items():
            facts = [to_fact_window(w, i) for i, w in enumerate(ws)]
            fired = fire(facts, self.config.rules, self._durations.get(video))
            self._fired.update({w.id: f for w, f in zip(ws, fired)})
        self.window_sec = float((self.header.get("facts") or {}).get("window_sec") or 4.0)
        self.groups = frozenset(w.group for w in self._windows)
        self._fits: dict[frozenset[str], tuple[dict[str, float], dict[str, tuple[int, int]], int, int]] = {}
        self._models: dict[frozenset[str], Model] = {}
        self._images: dict[str, str | None] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, directory: Path, config: ScoringConfig | None = None) -> "CaseBase":
        directory = Path(directory)
        data = json.loads((directory / BUNDLE_FILE).read_text(encoding="utf-8"))
        windows = [StoredWindow(
            id=w["id"], video=w["video"], group=w["group"], label=w["label"], start=float(w["start"]),
            end=float(w["end"]), relation=w["relation"], event_id=w.get("event_id"), type_id=w.get("type_id"),
            flags={k: bool(w.get("flags", {}).get(k)) for k in FLAG_KEYS}, people=int(w.get("people") or 0),
            scene_ru=w.get("scene_ru") or "", image=w.get("image"),
            frame_times=tuple(float(t) for t in w.get("frame_times") or ()),
        ) for w in data["windows"]]
        events = [StoredEvent(
            event_id=e["event_id"], video=e["video"], group=e["group"], t=float(e["t"]), type_id=e.get("type_id"),
            title=e.get("title") or "", description_ru=e.get("description_ru") or "", image=e.get("image"),
            flags={k: bool((e.get("flags") or {}).get(k)) for k in FLAG_KEYS}, people=int(e.get("people") or 0),
        ) for e in data.get("events", [])]
        durations = {v: float(d["duration"]) for v, d in (data.get("videos") or {}).items() if d.get("duration")}
        cb = cls(directory, windows, events, data.get("header"), config, durations)
        log.info("case base %s: %d windows, %d events, %d setups", directory, len(windows), len(events), len(cb.groups))
        return cb

    def configured(self, config: ScoringConfig) -> "CaseBase":
        """The same bundle scored with another configuration (fresh caches)."""
        return CaseBase(self.directory, self._windows, self._events, self.header, config, self._durations)

    # ---- learning (leave-one-setup-out safe) ----

    def _fit(self, groups: frozenset[str]) -> tuple[dict[str, float], dict[str, tuple[int, int]], int, int]:
        if groups in self._fits:
            return self._fits[groups]
        n_event = n_normal = 0
        hit_event: Counter[str] = Counter()
        hit_normal: Counter[str] = Counter()
        rule_type = {r.name: r.type_id for r in self.config.rules}
        for w in self._windows:
            if w.group not in groups:
                continue
            if w.relation == EVENT:
                n_event += 1
                hit_event.update(f for f in self._fired[w.id]
                                 if not (self.config.typed_hits and f in rule_type and rule_type[f] != w.type_id))
            elif w.relation == NORMAL:
                n_normal += 1
                hit_normal.update(self._fired[w.id])
        names = [r.name for r in self.config.rules] + list(UNUSUAL_FEATURES)
        if n_event == 0 or n_normal == 0:
            weights = {name: 0.0 for name in names}
        else:
            weights = {name: log_likelihood_ratio(hit_event[name], n_event, hit_normal[name], n_normal,
                                                  self.config.alpha) for name in names}
        hits = {name: (hit_event[name], hit_normal[name]) for name in names}
        self._fits[groups] = (weights, hits, n_event, n_normal)
        return self._fits[groups]

    def _nested_threshold(self, groups: frozenset[str]) -> float:
        """Each kept setup is scored with weights learned without it; the threshold is chosen on those scores."""
        video_scores = []
        for g in sorted(groups):
            weights = self._fit(groups - {g})[0]
            for ws in self._by_video.values():
                if ws[0].group != g:
                    continue
                violation = ws[0].label == "violation"
                best = 0.0
                for w in ws:
                    pairs = self._score(to_fact_window(w, 0), self._fired[w.id], weights).pairs(self.config.combine)
                    if violation and self.config.threshold_target == "useful":
                        pairs = [p for p in pairs if w.relation in (EVENT, NEAR)
                                 and (w.type_id is None or p[2] is None or p[2] == w.type_id)]
                    best = max([best] + [p[0] for p in pairs])
                video_scores.append((violation, best))
        th = choose_threshold(video_scores, self.config.recall_cost)
        if th is None:
            log.warning("no positive window scores on setups %s; using threshold 1.0", sorted(groups))
            return 1.0
        return th

    def model(self, exclude_groups: Iterable[str] = frozenset()) -> Model:
        key = frozenset(exclude_groups)
        with self._lock:
            if key not in self._models:
                groups = self.groups - key
                weights, hits, n_event, n_normal = self._fit(groups)
                threshold = self.config.threshold
                if threshold is None:
                    threshold = self._nested_threshold(groups)
                self._models[key] = Model(groups=groups, weights=weights, hits=hits, event_windows=n_event,
                                          normal_windows=n_normal, threshold=threshold)
                log.debug("model without %s: threshold %.3f", sorted(key), threshold)
            return self._models[key]

    # ---- scoring ----

    def _score(self, window: FactWindow, fired: frozenset[str], weights: dict[str, float]) -> WindowScore:
        cfg = self.config
        typed = sorted(((r.name, r.type_id, weights.get(r.name, 0.0)) for r in cfg.rules if r.name in fired),
                       key=lambda x: -x[2])
        unusual = sum(max(0.0, weights.get(f, 0.0)) for f in fired if f.startswith(UNUSUAL_PREFIX))
        if cfg.combine == "count":
            score, bonus = float(len(typed)), 0.0
        else:
            typed = [t for t in typed if t[2] > 0]
            bonus = min(cfg.bonus_cap, unusual)
            base = (typed[0][2] if typed else 0.0) if cfg.combine == "max" else sum(t[2] for t in typed)
            score = base + bonus
        if typed:
            rule, type_id = typed[0][0], typed[0][1]
        elif cfg.combine != "count" and unusual >= cfg.open_world_min:
            rule, type_id, score = UNUSUAL_RULE, None, unusual
        else:
            rule, type_id = None, None
        return WindowScore(window=window, score=score, rule=rule, type_id=type_id, typed=tuple(typed), bonus=bonus,
                           unusual=unusual, fired=fired)

    def score_windows(self, windows: Sequence[FactWindow], exclude_groups: Iterable[str] = frozenset(),
                      duration: float | None = None) -> list[WindowScore]:
        """Score every perceived window of one video (returned in time order; gaps allowed).

        `duration` is the true clip length: without it the last perceived window is taken as the end of the clip.
        """
        ordered = sorted(windows, key=lambda w: w.start)
        if ordered:
            lengths = sorted(w.end - w.start for w in ordered)
            if lengths[len(lengths) // 2] > WIDE_WINDOWS * self.window_sec:
                log.debug("windows of %.1f s vs %.1f s in the case base: weights overstate brief flags",
                          lengths[len(lengths) // 2], self.window_sec)
        weights = self.model(exclude_groups).weights
        fired = fire(ordered, self.config.rules, duration)
        return [self._score(w, f, weights) for w, f in zip(ordered, fired)]

    def candidates(self, windows: list[FactWindow], exclude_groups: frozenset[str] = frozenset(),
                   max_candidates: int = 2, duration: float | None = None) -> list[Candidate]:
        """Windows of ONE video (time order). Returns suspicious windows, best first; [] means 'looks clean'.

        Windows may have gaps (FactWindow.index is the true position, start/end the true times); pass the clip
        `duration` so that "until the end of the clip" rules fire only when the window covering the end was seen.
        A window may yield one candidate per counted rule type; candidates of the same type in touching windows are
        one episode and only the best is kept. Ties go to the earlier window (the onset).
        """
        threshold = self.model(exclude_groups).threshold
        pairs = [(score, ws.window, rule, type_id)
                 for ws in self.score_windows(windows, exclude_groups, duration)
                 for score, rule, type_id in ws.pairs(self.config.combine) if score >= threshold]
        pairs.sort(key=lambda p: (-p[0], p[1].start))
        chosen: list[Candidate] = []
        for score, window, rule, type_id in pairs:
            if len(chosen) >= max_candidates:
                break
            if any(c.type_id == type_id and _touch(c.window, window) for c in chosen):
                continue
            chosen.append(Candidate(window=window, type_id=type_id, rule=rule, score=round(score, 4)))
        return chosen

    # ---- references ----

    def image_b64(self, name: str | None) -> str | None:
        if not name:
            return None
        if name not in self._images:
            path = self.directory / name
            try:
                self._images[name] = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError:
                log.warning("case base image missing: %s", path)
                self._images[name] = None
        return self._images[name]

    def references(self, candidate: Candidate, exclude_groups: frozenset[str] = frozenset(),
                   k_violation: int = 1, k_normal: int = 1) -> list[CaseRef]:
        """Violation example of the same type (if type_id set) + the most similar labeled normal window."""
        excluded = frozenset(exclude_groups)
        w = candidate.window
        refs: list[CaseRef] = []
        if candidate.type_id is not None and k_violation > 0:
            events = [e for e in self._events if e.type_id == candidate.type_id and e.group not in excluded]
            events.sort(key=lambda e: (flag_distance(w.flags, w.people, e.flags, e.people), e.event_id))
            for e in self._with_images(events, k_violation, lambda e: e.image):
                refs.append(CaseRef(label="violation", type_id=e.type_id, image_b64=self.image_b64(e.image),
                                    description=e.description_ru, source=e.event_id, group=e.group))
        if k_normal > 0:
            normals = [s for s in self._windows if s.relation == NORMAL and s.group not in excluded]

            def distance(s: StoredWindow) -> float:
                same_rule = SAME_RULE_BONUS if candidate.rule in self._fired[s.id] else 0.0
                return flag_distance(w.flags, w.people, s.flags, s.people) - same_rule

            normals.sort(key=lambda s: (distance(s), s.id))
            for s in self._with_images(normals, k_normal, lambda s: s.image):
                refs.append(CaseRef(label="normal", type_id=None, image_b64=self.image_b64(s.image),
                                    description=s.scene_ru, source=s.id, group=s.group))
        return refs

    def _with_images(self, items: list, k: int, image_of) -> list:
        """First k items with a readable image, at most one per video."""
        out, videos = [], set()
        for item in items:
            if len(out) >= k:
                break
            if item.video in videos or self.image_b64(image_of(item)) is None:
                continue
            videos.add(item.video)
            out.append(item)
        return out

    # ---- inspection ----

    def videos(self) -> list[str]:
        return sorted(self._by_video, key=lambda v: (self._by_video[v][0].group, v))

    def video_info(self, video: str) -> dict[str, Any]:
        ws = self._by_video[video]
        events = sorted((e for e in self._events if e.video == video), key=lambda e: e.t)
        return {"video": video, "group": ws[0].group, "label": ws[0].label,
                "events": [{"event_id": e.event_id, "t": e.t, "type_id": e.type_id} for e in events]}

    def video_duration(self, video: str) -> float | None:
        return self._durations.get(video)

    def video_windows(self, video: str) -> list[FactWindow]:
        """Stored facts of a training video as the perception stage would deliver them (for offline replays)."""
        return [to_fact_window(w, i) for i, w in enumerate(self._by_video[video])]

    def stored_windows(self, video: str) -> list[StoredWindow]:
        return list(self._by_video[video])

    def summary(self) -> dict[str, Any]:
        model = self.model(frozenset())
        rules = [{"name": r.name, "type_id": r.type_id, "description": r.description,
                  "weight": round(model.weights[r.name], 3), "event_windows": model.hits[r.name][0],
                  "normal_windows": model.hits[r.name][1]} for r in self.config.rules]
        unusual = {f: {"weight": round(model.weights[f], 3), "event_windows": model.hits[f][0],
                       "normal_windows": model.hits[f][1]} for f in UNUSUAL_FEATURES}
        return {
            "directory": str(self.directory),
            "built": self.header.get("built"),
            "facts": self.header.get("facts"),
            "videos": len(self._by_video),
            "groups": len(self.groups),
            "windows": len(self._windows),
            "events": len(self._events),
            "relations": dict(Counter(w.relation for w in self._windows)),
            "event_types": dict(Counter(e.type_id or "new" for e in self._events)),
            "event_windows": model.event_windows,
            "normal_windows": model.normal_windows,
            "threshold": round(model.threshold, 4),
            "config": self.config.describe(),
            "rules": rules,
            "unusual": unusual,
        }
