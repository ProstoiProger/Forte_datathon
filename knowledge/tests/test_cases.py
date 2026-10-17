import numpy as np
import pytest

from forte_kb import config
from forte_kb.cases.evaluate import auc
from forte_kb.cases.labels import build_labels, canonical_id, parse_seconds
from forte_kb.cases.pipeline import state_changes
from forte_kb.cases.windows import build_windows, relation, window_starts
from forte_kb.store import Space


def test_parse_seconds():
    assert parse_seconds("57 секунд") == 57
    assert parse_seconds("1минута 8 секунд") == 68
    assert parse_seconds(None) is None
    assert parse_seconds("  ") is None


def test_canonical_id_maps_cyrillic_x():
    assert canonical_id("Х-24") == "X-24" == canonical_id("X-24")


@pytest.mark.skipif(not config.LABELS_XLSX.exists(), reason="organizers' data not present")
def test_real_labels_dedup():
    labels = build_labels(config.LABELS_XLSX, config.VIDEOS_DIR)
    vids = {v["video"]: v for v in labels["videos"]}
    assert len(vids) == 40
    assert sum(v["label"] == "violation" for v in vids.values()) == 21
    assert sum(len(v["events"]) for v in vids.values()) == 22
    assert [e["t"] for e in vids["X-4"]["events"]] == [57, 68] and vids["X-4"]["duplicates"] == ["X-47"]
    assert vids["X-5"]["duplicates"] == ["X-48"] and vids["X-5"]["label"] == "no_violation"


def test_window_starts_cover_end():
    assert window_starts(3.0, 4.0, 2.0) == [0.0]
    s = window_starts(11.6, 4.0, 2.0)
    assert s[0] == 0.0 and s[-1] == pytest.approx(7.6)


def test_relation():
    events = [{"t": 10.0, "event_id": "a"}]
    assert relation(8, 12, events, True)[0] == "event"
    assert relation(10, 14, events, True)[0] == "event"
    assert relation(11, 15, events, True)[0] == "near"
    assert relation(12.5, 16.5, events, True)[0] == "near"
    assert relation(20, 24, events, True)[0] == "other"
    assert relation(8, 12, [], False)[0] == "normal"


def test_build_windows_frames():
    frames = [{"t": i / 2, "file": f"t{i}.jpg"} for i in range(24)]
    video = {"video": "X-1", "group": "s01", "label": "violation",
             "events": [{"t": 5.0, "event_id": "X-1#e1", "type": "phone_use"}]}
    ws = build_windows(video, frames, 11.8, 4.0, 2.0)
    assert all(len(w["frames"]) == 8 for w in ws)
    ev = [w for w in ws if w["relation"] == "event"]
    assert {w["start"] for w in ev} == {2.0, 4.0} and ev[0]["event_type"] == "phone_use"


def test_space_search_and_filter(tmp_path):
    vecs = np.eye(3, dtype=np.float32)
    meta = [{"id": f"r{i}", "g": i % 2} for i in range(3)]
    sp = Space("t", vecs, meta)
    assert sp.search(np.array([1, 0.1, 0]), 1)[0]["id"] == "r0"
    assert sp.search(np.array([1, 0.1, 0]), 1, where=lambda m: m["g"] == 1)[0]["id"] == "r1"
    sp.save(tmp_path)
    assert Space.load(tmp_path, "t").search(np.array([0, 0, 1]), 1)[0]["id"] == "r2"


def test_state_changes():
    states = [{"t": 1.0, "safe_open": False}, {"t": 2.0, "safe_open": True}, {"t": 3.0, "safe_open": True}]
    ch = state_changes(states, 0, 5)
    assert [(c["t"], c["concept"], c["to"]) for c in ch] == [(2.0, "safe_open", True)]


def test_auc():
    assert auc([1, 2], [0]) == 1.0
    assert auc([0], [1]) == 0.0
    assert auc([1], [1]) == 0.5
    assert auc([], [1]) is None


def test_window_features_rules():
    from forte_kb.cases.facts_eval import BOOL, window_features

    def rec(**on):
        frame = {k: on.get(k, False) for k in BOOL}
        frame.update(frame=1, people=on.get("people", 1), cashier_at_workplace=on.get("cashier_at_workplace", True))
        return {"parsed": {"frames": [frame]}}

    recs = [rec(), rec(), rec(cashier_leaving=True, cash_on_desk=True, cashier_at_workplace=False)]
    X, names, rules = window_features(recs)
    assert X.shape == (3, len(names))
    assert rules[0] == 0 and rules[2] >= 3  # leaving, leaving with cash, cash unattended
    assert X[2, names.index("unusual:cashier_leaving")] == 1


def test_type_agreement_uses_reviewed_types():
    from forte_kb.cases.evaluate import type_agreement

    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    meta = [{"id": "a", "group": "s1", "relation": "event", "event_id": "A#e1", "event_type": "phone_use"},
            {"id": "b", "group": "s2", "relation": "event", "event_id": "B#e1", "event_type": "food_drink"},
            {"id": "c", "group": "s3", "relation": "event", "event_id": "C#e1", "event_type": "food_drink"}]
    sp = Space("t", vecs / np.linalg.norm(vecs, axis=1, keepdims=True), meta)
    assert type_agreement(sp)["nn_type_match"] < 1.0  # hypotheses: a=phone_use, its neighbour b=food_drink
    assert type_agreement(sp, {"A#e1": "food_drink"})["nn_type_match"] == 1.0  # review says a is food_drink
