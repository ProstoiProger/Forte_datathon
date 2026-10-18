import base64
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cashwatch.analyzers.facts.schema import FLAG_KEYS, Candidate, FactWindow
from cashwatch.casebase import BUNDLE_FILE, RULES, CaseBase, ScoringConfig
from cashwatch.casebase.evaluate import evaluate
from cashwatch.casebase.rules import fire
from cashwatch.casebase.store import StoredWindow, choose_threshold
from cashwatch.rules import load_registry

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_casebase", ROOT / "scripts" / "build_casebase.py")
build_casebase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_casebase)


def jpeg(path: Path, size: tuple[int, int] = (24, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", np.full((size[1], size[0], 3), 128, np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def flags(*on: str) -> dict[str, bool]:
    return {k: k in on for k in FLAG_KEYS}


def at(index: int, start: float, end: float, *on: str, people: int = 1) -> FactWindow:
    return FactWindow(index=index, start=start, end=end, frame_times=(), flags=flags(*on), people=people)


def facts(*windows: tuple[set[str], int]) -> list[FactWindow]:
    return [FactWindow(index=i, start=4.0 * i, end=4.0 * i + 4, frame_times=(), flags=flags(*on), people=people)
            for i, (on, people) in enumerate(windows)]


W = "cashier_at_workplace"
VIDEOS = {
    # video: (group, label, [(flags, people)], event (window index, type_id) or None)
    "V1": ("g1", "violation", [({W}, 1), ({W, "cashier_leaving", "cash_on_desk"}, 1), ({"cash_on_desk"}, 0)],
           (1, "cash_unsecured")),
    "V2": ("g1", "no_violation", [({W}, 1), ({W, "phone_in_hand"}, 1), ({W}, 1)], None),
    "V3": ("g2", "violation", [({W}, 1), ({W, "visitor_present"}, 3), ({W}, 1), ({W}, 1)], (1, "unauthorized_entry")),
    "V4": ("g2", "no_violation", [({W}, 1), ({W, "phone_in_hand"}, 1), ({W}, 1)], None),
    "V5": ("g3", "violation", [({W}, 1), ({W, "cashier_leaving", "cash_on_desk"}, 1), ({"cash_on_desk"}, 0)],
           (1, "cash_unsecured")),
    "V6": ("g3", "no_violation", [({W}, 1), ({W}, 1), ({W, "cash_in_hands"}, 1)], None),
}


def write_bundle(directory: Path, videos: dict = VIDEOS) -> Path:
    windows, events = [], []
    for video, (group, label, rows, event) in videos.items():
        for i, (on, people) in enumerate(rows):
            image = f"images/{video}_{i}.jpg"
            jpeg(directory / image)
            is_event = event is not None and event[0] == i
            relation = "normal" if label != "violation" else ("event" if is_event else "other")
            windows.append({
                "id": f"{video}@{4 * i:06.2f}", "video": video, "group": group, "label": label,
                "start": 4.0 * i, "end": 4.0 * i + 4, "relation": relation,
                "event_id": f"{video}#e1" if is_event else None, "type_id": event[1] if is_event else None,
                "flags": flags(*on), "people": people, "scene_ru": f"сцена {video} {i}", "image": image,
                "frame_times": [4.0 * i + 0.5 + k for k in range(4)],
            })
        if event is not None:
            i, type_id = event
            events.append({"event_id": f"{video}#e1", "video": video, "group": group, "t": 4.0 * i + 2,
                           "type_id": type_id, "title": type_id, "description_ru": f"нарушение {video}",
                           "image": f"images/{video}_{i}.jpg", "flags": flags(*rows[i][0]), "people": rows[i][1]})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BUNDLE_FILE).write_text(json.dumps({"header": {"built": "test"}, "windows": windows,
                                                     "events": events}, ensure_ascii=False), encoding="utf-8")
    return directory


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return write_bundle(tmp_path / "casebase")


def test_rule_types_exist_in_registry():
    registry = load_registry()
    assert len({r.name for r in RULES}) == len(RULES)
    for rule in RULES:
        assert rule.type_id is None or rule.type_id in registry, rule.name


def test_rules_use_context_and_durations():
    left = facts(({W}, 1), ({W, "cashier_leaving"}, 1), ({"door_open"}, 0))
    back = facts(({W}, 1), ({W, "cashier_leaving"}, 1), ({W}, 1))
    assert "left_workplace" in fire(left)[1]
    assert "left_workplace" not in fire(back)[1]

    items = facts(({W}, 1), ({W, "personal_items_on_desk"}, 1), ({W, "personal_items_on_desk"}, 1), ({W}, 1))
    assert "personal_items_long" in fire(items)[1]
    brief = facts(({W}, 1), ({W, "personal_items_on_desk"}, 1), ({W}, 1))
    assert "personal_items" in fire(brief)[1] and "personal_items_long" not in fire(brief)[1]
    bag = facts(({W, "personal_items_on_desk", "collection_bag"}, 1), ({W}, 1))
    assert "personal_items" not in fire(bag)[0]

    safe = facts(({W}, 1), ({W, "safe_open"}, 1), ({W, "safe_open"}, 1))
    assert "safe_open_at_end" in fire(safe)[2] and "safe_open_long" in fire(safe)[2]

    rare_visit = facts(({W}, 1), ({W}, 1), ({W, "visitor_present"}, 2), ({W}, 1))
    fired = fire(rare_visit)[2]
    assert "visitor_appears" in fired and "unusual:visitor_present" in fired and "unusual:people" in fired
    routine = facts(({W, "visitor_present"}, 2), ({W}, 1), ({W, "visitor_present"}, 2), ({W, "visitor_present"}, 2))
    assert "visitor_appears" not in fire(routine)[2]


def test_load_and_summary(bundle: Path):
    cb = CaseBase.load(bundle)
    s = cb.summary()
    assert (s["videos"], s["groups"], s["windows"], s["events"]) == (6, 3, 19, 3)
    assert s["event_windows"] == 3 and s["normal_windows"] == 9
    weights = {r["name"]: r for r in s["rules"]}
    assert weights["cash_unattended"]["weight"] > 0 and weights["cash_unattended"]["event_windows"] == 2
    assert weights["phone"]["weight"] < 0
    assert s["threshold"] > 0
    assert cb._images == {}  # images are read on first use


def test_statistics_are_leave_one_setup_out_safe(bundle: Path, tmp_path: Path):
    cb = CaseBase.load(bundle)
    without = {v: spec for v, spec in VIDEOS.items() if spec[0] != "g3"}
    reference = CaseBase.load(write_bundle(tmp_path / "without_g3", without))
    held_out, retrained = cb.model(frozenset({"g3"})), reference.model()
    assert held_out.weights == retrained.weights
    assert held_out.threshold == retrained.threshold
    assert cb.model(frozenset({"g3"})) is held_out  # cached per exclude set
    assert cb.model().weights != held_out.weights


def test_candidates_flag_violation_and_pass_clean(bundle: Path):
    cb = CaseBase.load(bundle)
    exclude = frozenset({"g3"})
    cands = cb.candidates(cb.video_windows("V5"), exclude)
    assert cands and isinstance(cands[0], Candidate)
    assert cands[0].type_id == "cash_unsecured" and cands[0].rule == "cash_unattended"
    assert cands[0].window.start == 4.0  # ties go to the onset
    assert all(c.type_id != cands[0].type_id or abs(c.window.index - cands[0].window.index) > 1 for c in cands[1:])
    assert cb.candidates(cb.video_windows("V6"), exclude) == []
    assert len(cb.candidates(cb.video_windows("V5"), exclude, max_candidates=1)) == 1


def test_open_world_candidate_without_typed_rule(bundle: Path):
    cb = CaseBase.load(bundle).configured(ScoringConfig(open_world_min=1.0, threshold=0.5))
    odd = facts(({W}, 1), ({W}, 1), ({W}, 2), ({W}, 1))  # more people than usual, no rule covers it
    cands = cb.candidates(odd, frozenset({"g3"}))
    assert [(c.type_id, c.rule, c.window.index) for c in cands] == [(None, "unusual", 2)]
    strict = CaseBase.load(bundle).configured(ScoringConfig(open_world_min=50.0, threshold=0.5))
    assert strict.candidates(odd, frozenset({"g3"})) == []


def test_references_same_type_other_setups_and_nearest_normal(bundle: Path):
    cb = CaseBase.load(bundle)
    exclude = frozenset({"g3"})
    cand = cb.candidates(cb.video_windows("V5"), exclude)[0]
    refs = cb.references(cand, exclude)
    assert [r.label for r in refs] == ["violation", "normal"]
    violation, normal = refs
    assert (violation.source, violation.type_id, violation.group) == ("V1#e1", "cash_unsecured", "g1")
    assert violation.description == "нарушение V1"
    assert normal.group != "g3" and normal.source.startswith(("V2@", "V4@")) and normal.description.startswith("сцена")
    assert base64.b64decode(violation.image_b64)[:2] == b"\xff\xd8"

    only_normal = cb.references(cand, frozenset({"g1", "g3"}))
    assert [r.label for r in only_normal] == ["normal"]
    assert cb.references(cand, frozenset({"g1", "g2", "g3"})) == []
    untyped = Candidate(window=cand.window, type_id=None, rule="unusual", score=1.0)
    assert [r.label for r in cb.references(untyped, exclude, k_normal=2)] == ["normal", "normal"]


def test_references_skip_missing_images(bundle: Path):
    (bundle / "images" / "V1_1.jpg").unlink()
    cb = CaseBase.load(bundle)
    cand = Candidate(window=cb.video_windows("V5")[1], type_id="cash_unsecured", rule="cash_unattended", score=2.0)
    assert [r.label for r in cb.references(cand, frozenset({"g3"}))] == ["normal"]


def test_choose_threshold_prefers_recall():
    scores = [(True, 3.0), (True, 1.0), (False, 1.0), (False, 0.0), (False, 0.0)]
    assert choose_threshold(scores, recall_cost=0.5) == pytest.approx(0.5)  # keep the violation at 1.0
    assert choose_threshold(scores, recall_cost=2.0) == pytest.approx(2.0)
    assert choose_threshold([(True, 0.0), (False, 0.0)], 0.5) is None


def test_evaluate_replay(bundle: Path):
    rep = evaluate(CaseBase.load(bundle))
    v = rep["videos"]
    assert v["tp"] + v["fn"] == 3 and v["fp"] + v["tn"] == 3
    assert rep["window_auc"] is not None and len(rep["rows"]) == 6
    assert rep["type"]["registry_events"] == 3


def test_stored_window_round_trip(bundle: Path):
    cb = CaseBase.load(bundle)
    stored = cb.stored_windows("V1")
    assert isinstance(stored[0], StoredWindow) and stored[1].relation == "event"
    replay = cb.video_windows("V1")
    assert [w.index for w in replay] == [0, 1, 2] and replay[1].flags["cash_on_desk"]


def write_store(root: Path) -> Path:
    """A tiny knowledge store in the layout of knowledge/store/cases."""
    labels = {"videos": [
        {"video": "X-1", "file": "X-1.mov", "group": "s01", "label": "violation",
         "events": [{"t": 6.0, "type": "phone_use", "event_id": "X-1#e1"}], "media": {"duration": 8.0}},
        {"video": "X-2", "file": "X-2.mov", "group": "s02", "label": "no_violation", "events": [],
         "media": {"duration": 4.0}},
    ]}
    (root / "vlm").mkdir(parents=True)
    (root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    cases = [
        {"video_id": "X-1", "group": "s01", "events": [{
            "event_id": "X-1#e1", "t": 6.0,
            "vlm": {"at_label_ru": "Кассир уходит.", "distinguishing_cue_ru": "Стол пуст.", "evidence_t": 7.0},
            "type": {"id": "workplace_abandoned", "title": "Кассир покинул рабочее место", "status": "confirmed",
                     "vlm_agrees": True}}]},
        {"video_id": "X-2", "group": "s02", "events": []},
    ]
    (root / "cases.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")

    def record(video: str, start: float, on: set[str], parsed: bool = True) -> str:
        frames = [{"frame": k + 1, "people": 1, **flags(*on)} for k in range(4)]
        return json.dumps({"key": f"{video}@{start:06.2f}", "video": video, "start": start, "end": start + 4,
                           "model": "qwen3.8:27b", "prompt_version": "v1",
                           "frame_times": [start + 0.5 + k for k in range(4)],
                           "parsed": {"frames": frames, "scene_ru": f"{video} {start}"} if parsed else None})

    lines = [record("X-1", 0.0, {"phone_in_hand"}), record("X-1", 0.0, {W}),  # the later record wins
             record("X-1", 4.0, {"cashier_leaving"}), record("X-2", 0.0, {W}, parsed=False)]
    (root / "vlm" / "facts.jsonl").write_text("\n".join(lines), encoding="utf-8")
    for video, seconds in (("X-1", 8.0), ("X-2", 4.0)):
        for ms in range(0, int(seconds * 1000), 500):
            jpeg(root / "frames" / video / f"t{ms:07d}.jpg", size=(848, 464))
    return root


def test_build_bundle_from_store(tmp_path: Path):
    store = write_store(tmp_path / "cases")
    out = tmp_path / "bundle"
    header = build_casebase.build(store / "labels.json", store / "cases.jsonl", store / "vlm" / "facts.jsonl",
                                  store / "frames", None, out)
    assert header["counts"]["windows"] == 2 and header["counts"]["events"] == 1  # X-2 had no parsed facts
    data = json.loads((out / BUNDLE_FILE).read_text(encoding="utf-8"))
    first, second = data["windows"]
    assert first["flags"][W] and not first["flags"]["phone_in_hand"]
    assert (first["relation"], second["relation"]) == ("near", "event")
    assert second["event_id"] == "X-1#e1" and second["type_id"] == "workplace_abandoned"
    assert first["image"] == "images/X-1_0002000.jpg"  # frame nearest the window middle
    (event,) = data["events"]
    assert event["image"] == "images/X-1_0007000.jpg"  # nearest the VLM evidence time
    assert event["description_ru"] == "Кассир уходит. Стол пуст." and event["window"] == "X-1@004.00"
    image = cv2.imread(str(out / first["image"]))
    assert max(image.shape[:2]) == 512
    provenance = {"model": "qwen3.8:27b", "prompt_version": "v1", "window_sec": 4.0, "frames_per_window": 4}
    assert header["facts"] == provenance
    assert data["videos"]["X-1"] == {"group": "s01", "label": "violation", "duration": 8.0}
    cb = CaseBase.load(out)
    assert cb.summary()["windows"] == 2 and cb.summary()["facts"] == provenance
    assert cb.video_duration("X-1") == 8.0


def test_gaps_end_of_clip_claims_need_the_end_window():
    leaves = [at(0, 0, 4, W), at(1, 4, 8, W, "cashier_leaving")]  # 8-16 not perceived
    assert "left_workplace" not in fire(leaves, duration=16.0)[1]
    assert "left_workplace" in fire(leaves)[1]  # without a duration the last perceived window is the end
    assert "left_workplace" in fire(leaves + [at(3, 12, 16, "door_open", people=0)], duration=16.0)[1]
    assert "left_workplace" not in fire(leaves + [at(2, 8, 12, W), at(3, 12, 16)], duration=16.0)[1]

    safe = [at(0, 0, 4, W), at(1, 4, 8, W, "safe_open")]
    assert "safe_open_at_end" not in fire(safe, duration=16.0)[1]
    assert "safe_open_at_end" in fire(safe + [at(3, 12, 16, W, "safe_open")], duration=16.0)[2]
    door = [at(0, 0, 4, W), at(1, 4, 8, W, "door_open")]
    assert "door_open_at_end" not in fire(door, duration=16.0)[1]
    assert "door_open_at_end" in fire(door, duration=8.2)[1]  # the window ends within the tolerance

    enters = [at(1, 4, 8, "cashier_leaving", "door_open"), at(2, 8, 12, W)]  # 0-4 not perceived
    assert "entered_door_open" not in fire(enters, duration=12.0)[0]
    assert "entered_door_open" in fire([at(0, 0, 4)] + enters, duration=12.0)[1]


def test_gaps_break_duration_runs():
    gap = [at(0, 0, 4, W, "safe_open"), at(2, 8, 12, W, "safe_open"), at(3, 12, 16, W)]
    assert all("safe_open_long" not in f for f in fire(gap, duration=16.0))
    run = [at(0, 0, 4, W, "safe_open"), at(1, 4, 8, W, "safe_open"), at(3, 12, 16, W)]
    assert "safe_open_long" in fire(run, duration=16.0)[0]
    nearly = [at(0, 0, 4, W, "drinking"), at(1, 4.3, 8.3, W, "drinking")]  # 0.3 s apart: still consecutive
    assert "drinking_long" in fire(nearly)[1]


def test_widened_windows_need_two_windows_for_a_duration():
    wide = [at(0, 0, 6.4, W), at(1, 6.4, 12.8, W, "drinking", "personal_items_on_desk"), at(2, 12.8, 19.2, W)]
    fired = fire(wide, duration=19.2)[1]
    assert "drinking_long" not in fired and "personal_items_long" not in fired and "personal_items" in fired
    two = [at(0, 0, 6.4, W, "drinking"), at(1, 6.4, 12.8, W, "drinking"), at(2, 12.8, 19.2, W)]
    assert "drinking_long" in fire(two, duration=19.2)[0]
    door = [at(0, 0, 10, W, "door_open"), at(1, 10, 20, W)]
    assert "door_open_long" not in fire(door, duration=20.0)[0]


def test_candidates_with_gaps_use_duration(bundle: Path):
    cb = CaseBase.load(bundle).configured(ScoringConfig(threshold=0.5))
    exclude = frozenset({"g3"})
    leaves = [at(0, 0, 4, W), at(1, 4, 8, W, "cashier_leaving")]
    assert all(c.type_id != "workplace_abandoned" for c in cb.candidates(leaves, exclude, duration=16.0))
    with_end = leaves + [at(3, 12, 16, people=0)]
    cands = cb.candidates(with_end, exclude, duration=16.0)
    assert [(c.type_id, c.window.index) for c in cands] == [("workplace_abandoned", 1)]

    items = [at(0, 0, 4, W), at(1, 4, 8, W, "personal_items_on_desk"), at(3, 12, 16, W, "personal_items_on_desk"),
             at(4, 16, 20, W)]
    cands = cb.candidates(items, exclude, duration=20.0)
    assert [(c.type_id, c.window.index) for c in cands] == [("personal_items", 1), ("personal_items", 3)]
    assert all(c.rule == "personal_items" for c in cands)  # separate episodes across the gap, neither is "long"
