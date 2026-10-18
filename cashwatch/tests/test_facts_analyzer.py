import time
from pathlib import Path

import pytest

from cashwatch.analyzers.base import AnalyzerError, Event, EvidenceRef
from cashwatch.analyzers.facts import FactsAnalyzer, plan_windows
from cashwatch.analyzers.facts.schema import FLAG_KEYS, OTHER_TYPE_ID, Candidate, FactWindow, Norm, Verification
from cashwatch.analyzers.vlm import Frame
from cashwatch.contract import validate_violation
from cashwatch.results import CONTRACT_KEYS, public_violation, render_violations
from cashwatch.rules import load_registry
from cashwatch.video import probe


def frames_for(duration: float, fps: float = 2.0) -> list[Frame]:
    return [Frame(t=round(i / fps, 3), image=f"img{i}") for i in range(int(duration * fps))]


def window(index: int, start: float, end: float, **flags) -> FactWindow:
    return FactWindow(index, start, end, (start,), {k: flags.get(k, False) for k in FLAG_KEYS}, flags.get("people", 1))


class FakePerceiver:
    def __init__(self, verdict: Verification | None = None, fail_facts: bool = False) -> None:
        self.verdict = verdict
        self.fail_facts = fail_facts
        self.facts_calls: list[int] = []
        self.verify_calls: list[dict] = []

    def facts(self, frames, index, start, end, deadline=None):
        self.facts_calls.append(index)
        return None if self.fail_facts else window(index, start, end, cashier_leaving=index == 2)

    def verify(self, frames, norms, refs, type_title, deadline=None):
        self.verify_calls.append({"frames": frames, "norms": norms, "refs": refs, "title": type_title, "deadline": deadline})
        return self.verdict


class FakeCaseBase:
    def __init__(self, candidates_for=None) -> None:
        self.candidates_for = candidates_for or (lambda windows: [])
        self.exclude_seen: list[frozenset] = []

    def candidates(self, windows, exclude_groups=frozenset(), max_candidates=2, duration=None):
        self.exclude_seen.append(exclude_groups)
        self.duration = duration
        return self.candidates_for(windows)[:max_candidates]

    def references(self, candidate, exclude_groups=frozenset(), k_violation=1, k_normal=1):
        return []


class FakeSearch:
    def __init__(self, norms: list[Norm]) -> None:
        self.norms = norms
        self.queries: list[str] = []

    def search(self, query, k=3):
        self.queries.append(query)
        return self.norms[:k]


def analyzer(perceiver, casebase, duration: float = 20.0, **kw) -> FactsAnalyzer:
    a = FactsAnalyzer(perceiver, casebase, load_registry(), describe_fn=lambda w: f"окно {w.index}", **kw)
    frames = frames_for(duration)
    a.load_frames = lambda path: (duration, frames)
    return a


def run(a: FactsAnalyzer, budget: float = 600.0) -> list[Event]:
    return a.analyze(Path("unused.mov"), lambda p: None, time.monotonic() + budget)


def leaving_candidate(windows, type_id="workplace_abandoned"):
    return [Candidate(w, type_id, "leaving", 2.0) for w in windows if w.flags["cashier_leaving"]]


def test_plan_windows_cover_video_and_widen():
    frames = frames_for(20.0)
    plan = plan_windows(frames, 20.0, 4.0, 4, 24)
    assert len(plan) == 5 and plan[0][0] == 0.0 and plan[-1][1] == 20.0
    assert all(len(w[2]) == 4 for w in plan)
    widened = plan_windows(frames, 20.0, 4.0, 4, 2)
    assert len(widened) == 2 and widened[0][1] == pytest.approx(10.0)


def test_clean_video_is_not_verified():
    p = FakePerceiver()
    events = run(analyzer(p, FakeCaseBase()))
    assert events == [] and p.verify_calls == [] and len(p.facts_calls) == 5


def test_registry_candidate_confirmed_uses_registry_norms_and_evidence_frame():
    p = FakePerceiver(Verification(True, 3, "Кассир выходит из помещения.", 0.9))
    a = analyzer(p, FakeCaseBase(leaving_candidate))
    events = run(a)
    call = p.verify_calls[0]
    assert call["title"].startswith(load_registry().get("workplace_abandoned").title) and "on camera:" in call["title"]
    assert call["norms"] and all(n.source == "registry" and n.quote for n in call["norms"])
    assert len(events) == 1 and events[0].type_id == "workplace_abandoned"
    assert events[0].timecode_sec == call["frames"][2].t
    assert events[0].observation == "Кассир выходит из помещения."


def test_rejected_candidate_gives_no_event():
    p = FakePerceiver(Verification(False, 0, "Обычная работа.", 0.8))
    assert run(analyzer(p, FakeCaseBase(leaving_candidate))) == []


def test_unusual_candidate_uses_search_and_keeps_chosen_norm():
    norms = [Norm("Акт 1", "п. 1, акт 1", "текст 1", source="rag"), Norm("Акт 2", "п. 2, акт 2", "текст 2", source="rag")]
    search = FakeSearch(norms)
    a = analyzer(FakePerceiver(), FakeCaseBase(lambda ws: leaving_candidate(ws, type_id=None)))
    p = FakePerceiver(Verification(True, 1, "Кассир кладёт купюры в карман.", 0.8, title="Наличные в одежде",
                                   norm_index=len(a.menu) + 2))
    a.perceiver = p
    a.regsearch = search
    events = run(a)
    assert search.queries == ["окно 2"]
    assert p.verify_calls[0]["title"] is None
    assert events[0].type_id == OTHER_TYPE_ID and events[0].title == "Наличные в одежде"
    assert events[0].norm["citation"] == "п. 2, акт 2" and events[0].norm["source"] == "rag"


def test_unusual_candidate_without_applicable_norm_is_dropped():
    p = FakePerceiver(Verification(True, 1, "Что-то необычное.", 0.8, norm_index=None))
    a = analyzer(p, FakeCaseBase(lambda ws: leaving_candidate(ws, type_id=None)))
    a.regsearch = FakeSearch([Norm("Акт", "п. 1", "текст", source="rag")])
    assert run(a) == []


def test_unusual_candidate_checked_against_registry_menu_first():
    a = analyzer(FakePerceiver(), FakeCaseBase(lambda ws: leaving_candidate(ws, type_id=None)))
    phone = next(i for i, (_, t) in enumerate(a.menu, 1) if t == "phone_use")
    p = FakePerceiver(Verification(True, 1, "Кассир держит телефон.", 0.9, norm_index=phone))
    a.perceiver = p
    events = run(a)
    norms = p.verify_calls[0]["norms"]
    assert len(norms) == len(a.menu) and all(n.source == "registry" for n in norms)
    assert events[0].type_id == "phone_use" and events[0].norm is None


def test_failed_verification_keeps_best_candidate_unverified():
    p = FakePerceiver(verdict=None)
    events = run(analyzer(p, FakeCaseBase(leaving_candidate), unverified_confidence=0.6))
    assert len(events) == 1 and events[0].confidence == 0.6 and events[0].type_id == "workplace_abandoned"


def test_exclude_groups_reach_case_base():
    cb = FakeCaseBase()
    a = analyzer(FakePerceiver(), cb)
    a.exclude_reference_groups = {"s09"}
    run(a)
    assert cb.exclude_seen == [frozenset({"s09"})]


def test_budget_caps_windows():
    p = FakePerceiver()
    a = analyzer(p, FakeCaseBase(), duration=60.0, est_call_sec=6.0, est_verify_sec=11.0, verify_reserve_sec=25.0, safety_sec=10.0)
    run(a, budget=10.0 + 11.0 + 6.0 * 3 + 1.0)  # 30 s available: reserve 11 s (one verification), 19 s -> 3 windows
    assert a.last_stats["allowed_windows"] == 3 and len(p.facts_calls) == 3


def test_reserve_scales_with_the_time_available():
    a = analyzer(FakePerceiver(), FakeCaseBase(), est_verify_sec=11.0, verify_reserve_sec=25.0)
    assert a.reserve_for(20.0) == 11.0 and a.reserve_for(50.0) == pytest.approx(15.0) and a.reserve_for(230.0) == 25.0


def test_motion_orders_perception():
    p = FakePerceiver()
    a = analyzer(p, FakeCaseBase(), motion_fn=lambda imgs: {"img0": 0.1, "img8": 0.9, "img16": 0.5}.get(imgs[0], 0.0))
    run(a)
    assert p.facts_calls[:4] == [0, 4, 1, 2]  # first and last windows, then by motion


def test_all_perception_failures_raise():
    with pytest.raises(AnalyzerError):
        run(analyzer(FakePerceiver(fail_facts=True), FakeCaseBase()))


def test_unlisted_type_renders_in_contract_format(video_path):
    info = probe(video_path)
    event = Event(OTHER_TYPE_ID, 1.0, 0.8, "Кассир кладёт купюры в карман.", [EvidenceRef(1.0)], title="Наличные в одежде",
                  norm={"title": "Акт", "citation": "п. 2, акт 2", "quote": "текст", "url": "", "source": "rag"})
    item = render_violations([event], video_path, info, load_registry())[0]
    assert item["type"] == "Наличные в одежде" and "подобрано поиском" in item["description"]
    assert item["rule"]["source"] == "rag" and item["rule"]["verified"] is False and item["rule"]["clause"] == "п. 2, акт 2"
    public = public_violation(item, detail=False)
    assert tuple(public) == CONTRACT_KEYS and validate_violation(public, "v", info.duration_sec) == []


def test_failed_verification_text_is_visible_facts_and_marked():
    events = run(analyzer(FakePerceiver(verdict=None), FakeCaseBase(leaving_candidate)))
    text = events[0].observation
    assert "кассир уходит с рабочего места" in text and "Не подтверждено" in text
    assert "посторон" not in text and "п. " not in text


def test_min_windows_are_perceived_even_when_reserve_is_needed():
    p = FakePerceiver()
    a = analyzer(p, FakeCaseBase(), duration=60.0, min_windows=3, safety_sec=10.0, verify_reserve_sec=25.0)
    run(a, budget=10.0 + 2.0)  # far less than the verification reserve
    assert len(p.facts_calls) == 3


def test_nothing_perceived_is_never_reported_clean():
    p = FakePerceiver()
    with pytest.raises(AnalyzerError):
        run(analyzer(p, FakeCaseBase(), safety_sec=10.0), budget=5.0)  # deadline already inside the safety margin
    assert p.facts_calls == []


def test_low_confidence_confirmation_is_kept_and_deadline_passed():
    p = FakePerceiver(Verification(True, 2, "Кассир уходит.", 0.2))
    a = analyzer(p, FakeCaseBase(leaving_candidate), unverified_confidence=0.6)
    events = run(a)
    assert events[0].confidence == 0.6 and p.verify_calls[0]["deadline"] is not None


def test_verification_frames_include_context_and_last_frame():
    p = FakePerceiver(Verification(False, 0, "", 0.1))
    run(analyzer(p, FakeCaseBase(leaving_candidate)))
    times = [f.t for f in p.verify_calls[0]["frames"]]
    assert times[0] <= 8.0 - 2.5 and times[-1] == 19.5  # window 2 is 8-12 s; the clip ends at 19.5 s


def test_duration_reaches_case_base():
    cb = FakeCaseBase()
    run(analyzer(FakePerceiver(), cb, duration=20.0))
    assert cb.duration == 20.0


def test_job_budget_shares_time_with_waiting_jobs():
    from cashwatch.jobs import job_budget
    assert job_budget(240.0, 0, True, 15.0) == 240.0
    assert job_budget(240.0, 4, True, 15.0) == 48.0
    assert job_budget(240.0, 4, False, 15.0) == 240.0
    assert job_budget(10.0, 4, True, 15.0) == 15.0


def test_empty_description_after_scrub_falls_back_to_visible_facts():
    p = FakePerceiver(Verification(True, 2, "", 0.9))
    events = run(analyzer(p, FakeCaseBase(leaving_candidate)))
    assert "кассир уходит с рабочего места" in events[0].observation


def test_rejected_candidate_of_recall_first_type_is_kept_and_marked():
    p = FakePerceiver(Verification(False, 0, "Кассир у двери.", 0.3))
    a = analyzer(p, FakeCaseBase(leaving_candidate), keep_rejected_types=frozenset({"workplace_abandoned"}))
    events = run(a)
    assert len(events) == 1 and events[0].confidence == a.unverified_confidence
    assert "не подтвердила" in events[0].observation
    assert run(analyzer(FakePerceiver(Verification(False, 0, "x", 0.3)), FakeCaseBase(leaving_candidate))) == []


def test_job_budget_cap_leaves_time_for_later_uploads():
    from cashwatch.jobs import job_budget
    assert job_budget(270.0, 0, True, 25.0, 100.0) == 100.0
    assert job_budget(270.0, 4, True, 25.0, 100.0) == 54.0
    assert job_budget(10.0, 4, True, 25.0, 100.0) == 25.0


def test_facts_without_stub_fallback_is_not_wrapped(make_settings, monkeypatch):
    import cashwatch.analyzers.facts as facts_module
    from cashwatch.analyzers import build_analyzer
    monkeypatch.setattr(facts_module, "build_facts_analyzer", lambda settings: analyzer(FakePerceiver(), FakeCaseBase()))
    assert isinstance(build_analyzer(make_settings(analyzer="facts", fallback="none")), FactsAnalyzer)
