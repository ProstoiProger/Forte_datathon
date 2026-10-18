"""Facts analyzer: VLM facts -> case-base candidates -> VLM verification against a norm.

Per video:
  1. Frames at `fps`, non-overlapping windows of `window_sec` with `frames_per_window` frames each. The number of
     windows is capped by the job's deadline, which the job manager shares fairly with the jobs waiting in the queue,
     so a long queue makes each video cheaper instead of late.
  2. Perception: one compact VLM call per window -> FactWindow. Windows with more motion go first, so when time runs
     out the skipped windows are the static ones. At least `min_windows` are perceived even if the verification
     reserve has to be spent; a video is never declared clean without being looked at.
  3. Case base: rules weighted by statistics of normal work -> up to `max_candidates` suspicious windows.
  4. Verification: for the best `max_verify` candidates, one VLM call with the candidate frames (window +- context and
     the last frame of the clip), reference examples and the norm. A known type is checked against its verified
     registry citations. An unusual situation is checked against a menu: one verified citation per registry type,
     then the clauses found by regulation search.
  5. Confirmed candidates become events. If nothing is confirmed and a typed candidate could not be verified (failed
     call or no time left), it is kept with `unverified_confidence` and marked as unverified.
The VLM never writes legal text: citations come from the registry or the regulation index and are attached by code.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from cashwatch.analyzers.base import AnalyzerError, Event, EvidenceRef, ProgressFn
from cashwatch.analyzers.facts.schema import OTHER_TYPE_ID, Candidate, CaseRef, FactWindow, Norm, Verification
from cashwatch.analyzers.vlm import Frame, merge_events, sample_frames
from cashwatch.rules import Registry, ViolationType, load_registry
from cashwatch.video import probe

log = logging.getLogger("cashwatch.facts")

MENU_QUOTE_CHARS = 300
# What each registry type looks like on camera. The verifier is asked about these visible signs; the legal title and
# citations in the output still come from the registry. Without them the model reads a title such as "left the
# workplace during service" literally and rejects clips with no client, which is how the training data labels them.
VISUAL_DEFINITIONS = {
    "workplace_abandoned": "the cashier walks away from the workplace or out of the room while cash, documents, a bag "
                           "or working equipment (for example an unlocked computer) stay on the desk unattended",
    "cash_unsecured": "cash or cash bundles stay on the desk or outside a locked safe or drawer while the cashier is "
                      "away or busy elsewhere",
    "safe_left_open": "the safe door stays open while the safe is not being used (the cashier sits at the desk or "
                      "walks away)",
    "door_left_open": "the cash room door stays open after people pass through it while the cashier works inside",
    "personal_items": "a coat, handbag or other personal belongings are put on the cash desk",
    "food_drink": "the cashier eats or drinks at the cash workplace",
    "phone_use": "the cashier uses a mobile phone at the cash workplace",
    "unauthorized_entry": "a person who is neither the cashier nor a client being served across the desk enters or "
                          "stays in the cash room, for example next to the cashier behind the desk",
    "collector_handover": "a cash-collection bag is handed over or opened without the collector staying for the whole "
                          "acceptance",
    "cash_handling": "cash is counted or handled at the desk after the client has left, not in the client's presence",
    "no_dual_control": "cash is counted or packed by the cashier alone where a second, controlling employee is needed",
    "keys_unattended": "safe keys are left unattended on the desk or in the lock",
    "authenticity_check_skipped": "banknotes are accepted or counted by hand without putting them through the "
                                  "banknote checking device",
    "camera_obstruction": "cash is handled out of the camera's view or the view is blocked",
    "cash_concealment": "banknotes are moved into clothing, pockets or personal bags",
}
UNVERIFIED_NOTE = "Не подтверждено повторной проверкой модели."
REJECTED_NOTE = "Повторная проверка моделью не подтвердила; событие оставлено, так как пропуск нарушения недопустим."
VISIBLE_RU = {
    "cashier_leaving": "кассир уходит с рабочего места",
    "cash_on_desk": "на столе лежат наличные",
    "cash_in_hands": "у кассира в руках наличные",
    "safe_open": "дверца сейфа открыта",
    "door_open": "дверь помещения открыта",
    "phone_in_hand": "у кассира в руке телефон",
    "drinking": "кассир держит напиток",
    "personal_items_on_desk": "на столе личные вещи",
    "collection_bag": "видна инкассаторская сумка",
    "visitor_present": "в помещении есть посетитель",
}


class PerceiverLike(Protocol):
    def facts(self, frames: list[Frame], index: int, start: float, end: float, deadline: float | None = None) -> FactWindow | None: ...
    def verify(self, frames: list[Frame], norms: list[Norm], refs: list[CaseRef], type_title: str | None,
               deadline: float | None = None) -> Verification | None: ...


class CaseBaseLike(Protocol):
    def candidates(self, windows: list[FactWindow], exclude_groups: frozenset[str] = frozenset(), max_candidates: int = 2,
                   duration: float | None = None) -> list[Candidate]: ...
    def references(self, candidate: Candidate, exclude_groups: frozenset[str] = frozenset(), k_violation: int = 1, k_normal: int = 1) -> list[CaseRef]: ...


class RegSearchLike(Protocol):
    def search(self, query: str, k: int = 3) -> list[Norm]: ...


def plan_windows(frames: list[Frame], duration: float, window_sec: float, per_window: int, max_windows: int) -> list[tuple[float, float, list[Frame]]]:
    """Non-overlapping windows covering the video; widened when more than `max_windows` would be needed."""
    if not frames:
        return []
    duration = max(duration, frames[-1].t)
    if math.ceil(duration / window_sec) > max_windows:
        window_sec = duration / max_windows
    out = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + window_sec, duration)
        inside = [f for f in frames if start - 1e-6 <= f.t < end - 1e-6] or ([frames[-1]] if end >= duration else [])
        if inside:
            out.append((start, end, evenly(inside, per_window)))
        start += window_sec
    return out


def evenly(items: list, n: int, offset: float = 0.0) -> list:
    """n items spread evenly; `offset` (0..1 of a step) picks the in-between items for a second, different sample."""
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1) if n > 1 else 0
    return [items[min(len(items) - 1, round(k * step + offset * step))] for k in range(n)]


def visible_facts_ru(window: FactWindow) -> str:
    """Plain description of the perceived states, without legal wording."""
    states = [text for key, text in VISIBLE_RU.items() if window.flags.get(key)]
    if not window.flags.get("cashier_at_workplace") and not window.flags.get("cashier_leaving"):
        states.insert(0, "кассира нет на рабочем месте")
    head = f"{window.start:.0f}–{window.end:.0f} с: людей в помещении — {window.people}"
    return head + ("; " + "; ".join(states) if states else "") + "."


def norm_from_citation(vtype: ViolationType, quote_chars: int | None = None) -> list[Norm]:
    return [
        Norm(title=vtype.title, citation=f"{c.clause}, {c.act.short}",
             quote=c.quote if quote_chars is None else c.quote[:quote_chars], url=c.act.url, source="registry")
        for c in vtype.verified_citations
    ]


class FactsAnalyzer:
    name = "facts"
    default_threshold = 0.5

    def __init__(
        self,
        perceiver: PerceiverLike,
        casebase: CaseBaseLike,
        registry: Registry | None = None,
        regsearch: RegSearchLike | None = None,
        describe_fn: Callable[[FactWindow], str] | None = None,
        motion_fn: Callable[[list[str]], float] | None = None,
        fps: float = 2.0,
        window_sec: float = 4.0,
        frames_per_window: int = 4,
        verify_frames: int = 8,
        verify_context_sec: float = 3.0,
        max_windows: int = 24,
        min_windows: int = 3,
        max_candidates: int = 2,
        max_verify: int = 2,
        max_norms: int = 2,
        search_k: int = 3,
        max_side: int = 848,
        jpeg_quality: int = 85,
        est_call_sec: float = 6.0,
        est_verify_sec: float = 11.0,
        verify_reserve_sec: float = 25.0,
        safety_sec: float = 10.0,
        merge_gap_sec: float = 4.0,
        unverified_confidence: float = 0.6,
        keep_rejected_types: frozenset[str] = frozenset(),
        double_check_types: frozenset[str] = frozenset(),
    ) -> None:
        self.perceiver = perceiver
        self.casebase = casebase
        self.registry = registry or load_registry()
        self.regsearch = regsearch
        self.describe_fn = describe_fn
        self.motion_fn = motion_fn
        self.fps = fps
        self.window_sec = window_sec
        self.frames_per_window = frames_per_window
        self.verify_frames = verify_frames
        self.verify_context_sec = verify_context_sec
        self.max_windows = max_windows
        self.min_windows = min_windows
        self.max_candidates = max_candidates
        self.max_verify = max_verify
        self.max_norms = max_norms
        self.search_k = search_k
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.est_call_sec = est_call_sec
        self.est_verify_sec = est_verify_sec
        self.verify_reserve_sec = verify_reserve_sec
        self.safety_sec = safety_sec
        self.merge_gap_sec = merge_gap_sec
        self.unverified_confidence = unverified_confidence
        self.keep_rejected_types = frozenset(keep_rejected_types)  # recall over the verifier's veto for these types
        # types whose confirmation must hold on a second, shifted set of frames ("unusual" = candidates without a type)
        self.double_check_types = frozenset(double_check_types)
        self.menu = [(n, t.type_id) for t in self.registry.all() for n in norm_from_citation(t, MENU_QUOTE_CHARS)[:1]]
        self.exclude_reference_groups: set[str] = set()  # set by the evaluation harness to the scored video's setup
        self.last_stats: dict[str, Any] = {}

    def check(self) -> tuple[bool, str]:
        client = getattr(self.perceiver, "client", None)
        check = getattr(client, "check", None)
        ok, detail = check() if check else (True, "perceiver without check")
        search_check = getattr(self.regsearch, "check", None)
        if search_check:
            search_ok, search_detail = search_check()
            detail += f"; {search_detail}" if search_ok else f"; regulation search unavailable ({search_detail})"
        return ok, detail

    def load_frames(self, video_path: Path) -> tuple[float, list[Frame]]:
        info = probe(video_path)
        return info.duration_sec, sample_frames(video_path, self.fps, self.max_side, self.jpeg_quality)

    def analyze(self, video_path: Path, progress: ProgressFn, deadline: float) -> list[Event]:
        started = time.monotonic()
        duration, frames = self.load_frames(video_path)
        available = deadline - time.monotonic() - self.safety_sec
        reserve = self.reserve_for(available)
        allowed = max(self.min_windows, min(self.max_windows, int(max(available - reserve, 0.0) // self.est_call_sec)))
        windows = plan_windows(frames, duration, self.window_sec, self.frames_per_window, allowed)
        stats: dict[str, Any] = {"frames": len(frames), "windows": len(windows), "allowed_windows": allowed,
                                 "verify_reserve_sec": round(reserve, 1)}

        facts = self._perceive(windows, progress, deadline, reserve, stats)
        if not facts:
            raise AnalyzerError(f"no window perceived (windows={len(windows)}, failed={stats['facts_failed']}, "
                                f"skipped={stats['facts_skipped']})")
        exclude = frozenset(self.exclude_reference_groups)
        candidates = self.casebase.candidates(facts, exclude, self.max_candidates, duration=duration)
        stats["candidates"] = [(c.window.index, c.type_id, c.rule, round(c.score, 3)) for c in candidates]
        events = self._verify(candidates, frames, exclude, deadline, stats)
        stats["seconds"] = round(time.monotonic() - started, 2)
        self.last_stats = stats
        progress(1.0)
        return merge_events(events, self.merge_gap_sec)

    def reserve_for(self, available: float) -> float:
        """Time kept for verification: at least one verification call, at most `verify_reserve_sec`, about 30 % of
        the time available, so a short fair share in a queue still leaves most of it for perception."""
        return min(self.verify_reserve_sec, max(self.est_verify_sec, 0.3 * available))

    def _perceive(self, windows: list[tuple[float, float, list[Frame]]], progress: ProgressFn, deadline: float, reserve_sec: float,
                  stats: dict) -> list[FactWindow]:
        order = list(range(len(windows)))
        if self.motion_fn:
            motion = [self.motion_fn([f.image for f in w[2]]) for w in windows]
            order.sort(key=lambda i: -motion[i])
        # the first and last windows go first: rules about entering, leaving and states left at the end need them,
        # and after the cashier leaves the room is static, so motion alone would skip the end
        ends = [i for i in dict.fromkeys((0, len(windows) - 1)) if 0 <= i < len(windows)]
        order = ends + [i for i in order if i not in ends]
        facts: list[FactWindow] = []
        failed = skipped = 0
        t0 = time.monotonic()
        for n, i in enumerate(order, 1):
            reserve = reserve_sec if len(facts) >= self.min_windows else 0.0
            if time.monotonic() > deadline - self.safety_sec - reserve:
                skipped = len(order) - n + 1
                log.warning("perception stopped by the deadline: %d of %d windows skipped", skipped, len(order))
                break
            start, end, wframes = windows[i]
            fw = self.perceiver.facts(wframes, i, start, end, deadline=deadline - self.safety_sec)
            if fw is None:
                failed += 1
            else:
                facts.append(fw)
            progress(0.8 * n / len(order))
        stats.update(perceived=len(facts), facts_failed=failed, facts_skipped=skipped, perception_sec=round(time.monotonic() - t0, 2))
        return sorted(facts, key=lambda w: w.index)

    def _verify(self, candidates: list[Candidate], frames: list[Frame], exclude: frozenset[str], deadline: float, stats: dict) -> list[Event]:
        events: list[Event] = []
        unverified: Candidate | None = None
        kept_rejected: Candidate | None = None
        verified = rejected = 0
        t0 = time.monotonic()
        for cand in candidates[: self.max_verify]:
            typed = bool(cand.type_id and cand.type_id in self.registry)
            if time.monotonic() > deadline - self.safety_sec:
                log.warning("no time left to verify candidate %d", cand.window.index)
                unverified = unverified or (cand if typed else None)
                break
            vframes = self._candidate_frames(cand.window, frames)
            norms, norm_types, title = self._norms(cand)
            if not norms:
                stats.setdefault("skipped_no_norm", []).append(cand.window.index)
                continue
            refs = self.casebase.references(cand, exclude)
            v = self.perceiver.verify(vframes, norms, refs, title, deadline=deadline - self.safety_sec)
            stats.setdefault("verifications", []).append({
                "window": cand.window.index, "type_id": cand.type_id, "rule": cand.rule, "norms": len(norms),
                "refs": [r.source for r in refs], "confirmed": None if v is None else v.confirmed,
                "frame_t": vframes[v.frame - 1].t if v and v.frame else None,
                "norm_index": None if v is None else v.norm_index, "description": "" if v is None else v.description[:160],
            })
            if v is None:
                unverified = unverified or (cand if typed else None)
                continue
            verified += 1
            if v.confirmed and self._needs_double_check(cand):
                second = self.perceiver.verify(self._candidate_frames(cand.window, frames, offset=0.5), norms, refs, title,
                                               deadline=deadline - self.safety_sec)
                # agreement = the second sample also confirms; the norm stays the first pass's choice (on the
                # training videos the two passes often picked neighbouring clauses for the same event)
                stats["verifications"][-1]["second"] = None if second is None else second.confirmed
                stats["verifications"][-1]["second_norm_index"] = None if second is None else second.norm_index
                if second is not None and not second.confirmed:
                    stats["double_check_rejected"] = stats.get("double_check_rejected", 0) + 1
                    v = Verification(False, 0, v.description, v.confidence)
            if not v.confirmed:
                rejected += 1
                if cand.type_id in self.keep_rejected_types:
                    kept_rejected = kept_rejected or cand
                continue
            event = self._event(cand, v, vframes, norms, norm_types)
            if event is not None:
                events.append(event)
        confirmed = len(events)
        if not events and unverified is not None:
            events.append(self._unverified(unverified, self._candidate_frames(unverified.window, frames)))
        elif not events and kept_rejected is not None:
            events.append(self._unverified(kept_rejected, self._candidate_frames(kept_rejected.window, frames), REJECTED_NOTE))
        stats.update(verified=verified, rejected=rejected, confirmed=confirmed, unverified_kept=len(events) - confirmed,
                     verify_sec=round(time.monotonic() - t0, 2))
        return events

    def _needs_double_check(self, cand: Candidate) -> bool:
        return (cand.type_id or "unusual") in self.double_check_types

    def _candidate_frames(self, window: FactWindow, frames: list[Frame], offset: float = 0.0) -> list[Frame]:
        """The window with `verify_context_sec` on each side, evenly thinned, plus the last frame of the clip so the
        verifier can judge whether a state persists (a door or a safe left open, the cashier not returning)."""
        lo, hi = window.start - self.verify_context_sec, window.end + self.verify_context_sec
        chosen = evenly([f for f in frames if lo <= f.t <= hi] or frames[:1], self.verify_frames, offset)
        if frames and frames[-1].t > chosen[-1].t:
            chosen.append(frames[-1])
        return chosen

    def _norms(self, cand: Candidate) -> tuple[list[Norm], list[str | None], str | None]:
        """Norms to check, the registry type of each (None for searched clauses), and the type title if known."""
        if cand.type_id and cand.type_id in self.registry:
            vtype = self.registry.get(cand.type_id)
            norms = norm_from_citation(vtype)[: self.max_norms]
            definition = VISUAL_DEFINITIONS.get(vtype.type_id)
            title = f"{vtype.title} — on camera: {definition}" if definition else vtype.title
            return norms, [vtype.type_id] * len(norms), title
        norms = [n for n, _ in self.menu]
        types: list[str | None] = [t for _, t in self.menu]
        if self.regsearch is not None and self.describe_fn is not None:
            found = self.regsearch.search(self.describe_fn(cand.window), k=self.search_k)
            norms += found
            types += [None] * len(found)
        return norms, types, None

    def _event(self, cand: Candidate, v: Verification, vframes: list[Frame], norms: list[Norm], norm_types: list[str | None]) -> Event | None:
        k = min(max(v.frame, 1), len(vframes)) - 1
        common = dict(
            timecode_sec=vframes[k].t,
            confidence=max(0.0, min(1.0, max(float(v.confidence), self.unverified_confidence))),
            # the description can be empty after legal references were removed from the model's text
            observation=(v.description.strip() or visible_facts_ru(cand.window))[:400],
            evidence=[EvidenceRef(vframes[j].t) for j in (k - 1, k, k + 1) if 0 <= j < len(vframes)],
            raw_confidence=round(cand.score, 4),
            window=cand.window.index,
        )
        if cand.type_id and cand.type_id in self.registry:
            return Event(type_id=cand.type_id, **common)
        if not v.norm_index or not 1 <= v.norm_index <= len(norms):
            log.info("unusual situation in window %d confirmed without an applicable norm; dropped", cand.window.index)
            return None
        chosen_type = norm_types[v.norm_index - 1]
        if chosen_type is not None:
            return Event(type_id=chosen_type, **common)
        return Event(type_id=OTHER_TYPE_ID, title=v.title, norm=asdict(norms[v.norm_index - 1]), **common)

    def _unverified(self, cand: Candidate, vframes: list[Frame], note: str = UNVERIFIED_NOTE) -> Event:
        k = len(vframes) // 2
        return Event(
            type_id=str(cand.type_id),
            timecode_sec=vframes[k].t,
            confidence=self.unverified_confidence,
            observation=f"{visible_facts_ru(cand.window)} {note}",
            evidence=[EvidenceRef(vframes[j].t) for j in (k - 1, k, k + 1) if 0 <= j < len(vframes)],
            raw_confidence=round(cand.score, 4),
            window=cand.window.index,
        )


def _casebase_path(configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute() or path.exists():
        return path
    beside_project = Path(__file__).resolve().parents[4] / path  # relative to the cashwatch folder
    return beside_project if beside_project.exists() else path


def build_facts_analyzer(settings: Any) -> FactsAnalyzer:
    from cashwatch.analyzers.facts.perception import Perceiver, window_motion
    from cashwatch.analyzers.vlm import OllamaClient
    from cashwatch.casebase import CaseBase
    from cashwatch.regsearch import RegSearchClient, describe_facts_ru

    casebase_dir = _casebase_path(settings.casebase_dir)
    if not (casebase_dir / "casebase.json").exists():
        raise FileNotFoundError(f"case base not found at {casebase_dir}; build it with `uv run python scripts/build_casebase.py`")
    client = OllamaClient(
        settings.vlm_url,
        settings.vlm_model,
        timeout=settings.vlm_timeout_sec,
        num_ctx=settings.vlm_num_ctx,
        keep_alive=settings.vlm_keep_alive,
    )
    regsearch = RegSearchClient(settings.regsearch_url, settings.regsearch_timeout_sec) if settings.regsearch_url else None
    return FactsAnalyzer(
        Perceiver(client, retries=settings.vlm_retries),
        CaseBase.load(casebase_dir),
        load_registry(),
        regsearch,
        describe_fn=describe_facts_ru,
        motion_fn=window_motion,
        fps=settings.vlm_fps,
        window_sec=settings.facts_window_sec,
        frames_per_window=settings.facts_frames,
        verify_frames=settings.facts_verify_frames,
        verify_context_sec=settings.facts_verify_context_sec,
        max_windows=settings.facts_max_windows,
        min_windows=settings.facts_min_windows,
        max_candidates=settings.facts_max_candidates,
        max_verify=settings.facts_max_verify,
        max_side=settings.vlm_max_side,
        jpeg_quality=settings.vlm_jpeg_quality,
        est_call_sec=settings.facts_est_call_sec,
        est_verify_sec=settings.facts_est_verify_sec,
        verify_reserve_sec=settings.facts_verify_reserve_sec,
        safety_sec=settings.vlm_safety_sec,
        merge_gap_sec=settings.vlm_merge_gap_sec,
        unverified_confidence=settings.facts_unverified_confidence,
        keep_rejected_types=frozenset(t.strip() for t in settings.facts_keep_rejected_types.split(",") if t.strip()),
        double_check_types=frozenset(t.strip() for t in settings.facts_double_check_types.split(",") if t.strip()),
    )
