import importlib.util
import json
from pathlib import Path

import pytest

from cashwatch.evaluation import Pred, Row, build_rows, holdout, logo_cv, match, pick_tau, score, video_report
from conftest import write_video

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows() -> list[Row]:
    return [
        Row("v1", "a", True, [(10.0, "unauthorized_entry")], [Pred(10.5, "unauthorized_entry", 0.9)], 3.0),
        Row("v2", "a", True, [(5.0, "phone_use")], [Pred(9.0, "food_drink", 0.6)], 4.0),
        Row("v3", "b", False, [], [Pred(2.0, "phone_use", 0.55)], 5.0),
        Row("v4", "b", False, [], [], 6.0),
        Row("v5", "b", True, [(3.0, "safe_left_open"), (20.0, "workplace_abandoned")], [Pred(3.0, "safe_left_open", 0.7)], 2.0),
    ]


def test_detection_counts_at_threshold() -> None:
    s = score(rows(), 0.5)
    d = s["detection"]
    assert (d["tp"], d["fn"], d["fp"], d["tn"]) == (3, 0, 1, 1)
    assert d["recall"] == 1.0 and d["fpr"] == 0.5 and d["accuracy"] == pytest.approx(0.8)
    s = score(rows(), 0.65)
    assert (s["detection"]["tp"], s["detection"]["fp"]) == (2, 0)


def test_localization_and_type_accuracy() -> None:
    s = score(rows(), 0.5, tol=2.0, match_window=5.0)
    loc, cls = s["localization"], s["classification"]
    assert loc["events"] == 4 and loc["matched"] == 3
    assert loc["within_tol"] == pytest.approx(2 / 4)
    assert loc["mae_sec"] == pytest.approx((0.5 + 4.0 + 0.0) / 3)
    assert cls["accuracy"] == pytest.approx(2 / 4)
    assert cls["accuracy_matched"] == pytest.approx(2 / 3)
    assert s["proxy_55"] == pytest.approx(25 * 0.8 + 15 * 0.5 + 15 * 0.5)
    assert s["latency"]["max"] == 6.0


def test_match_is_one_to_one_and_respects_window() -> None:
    preds = [Pred(10.0, "x", 1.0)]
    assert [p is not None for _, p in match([(9.0, None), (11.5, None)], preds, 5.0)] == [True, False]
    assert match([(30.0, None)], preds, 5.0) == [(0, None)]


def test_pick_tau_keeps_recall_target() -> None:
    assert pick_tau(rows(), min_recall=1.0) <= 0.6
    assert pick_tau(rows(), min_recall=0.6) >= 0.6


def test_logo_cv_tunes_without_the_held_out_group() -> None:
    data = [
        Row("a1", "a", True, [(1.0, None)], [Pred(1.0, "x", 0.3)]),
        Row("a2", "a", False, [], [Pred(1.0, "x", 0.2)]),
        Row("b1", "b", True, [(1.0, None)], [Pred(1.0, "x", 0.9)]),
        Row("b2", "b", False, [], [Pred(1.0, "x", 0.8)]),
    ]
    cv = logo_cv(data, min_recall=1.0)
    assert cv["tau_by_group"]["a"] > 0.8
    assert cv["tau_by_group"]["b"] <= 0.3
    assert (cv["detection"]["tp"], cv["detection"]["fn"], cv["detection"]["fp"], cv["detection"]["tn"]) == (1, 1, 1, 1)
    assert score(data, 0.25)["detection"]["recall"] == 1.0


def test_holdout_uses_only_other_groups_for_tau() -> None:
    result = holdout(rows(), {"b"}, min_recall=1.0)
    assert result["train_videos"] == 2 and result["detection"]["videos"] == 3


def test_video_report_flags_issues() -> None:
    report = {r["video"]: r for r in video_report(rows(), 0.5)}
    assert "type food_drink != phone_use" in report["v2"]["issues"]
    assert "off by +4.0s" in report["v2"]["issues"]
    assert report["v3"]["issues"] == "false_positive"
    assert "event@20s unmatched" in report["v5"]["issues"]
    assert report["v4"]["issues"] == ""


def test_build_rows_joins_labels_and_predictions() -> None:
    labels = {"videos": [
        {"video": "X-1", "group": "s01", "label": "violation", "events": [{"t": 22.0, "type": "cash_handling"}]},
        {"video": "X-3", "group": "s01", "label": "no_violation", "events": []},
        {"video": "X-9", "group": "s02", "label": "no_violation", "events": []},
    ]}
    preds = [
        {"video": "X-1", "events": [{"t": 21.0, "type": "cash_handling", "conf": 0.7}], "seconds": 1.5},
        {"video": "X-3", "events": [], "seconds": 1.0, "error": None},
    ]
    built = build_rows(labels, preds)
    assert [r.video for r in built] == ["X-1", "X-3"]
    assert built[0].gt == [(22.0, "cash_handling")] and built[0].preds[0].conf == 0.7


def test_evaluate_cli_runs_stub_and_scores(tmp_path: Path, capsys) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    write_video(videos / "a.mov", seconds=4.0)
    write_video(videos / "b.mov", seconds=4.0)
    labels = {"videos": [
        {"video": "A", "file": "a.mov", "group": "s01", "label": "violation", "events": [{"t": 2.5, "type": None}]},
        {"video": "B", "file": "b.mov", "group": "s02", "label": "no_violation", "events": []},
    ]}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    evaluate = load_script("evaluate")
    common = ["--labels", str(labels_path), "--videos", str(videos), "--runs-dir", str(tmp_path / "runs"), "--analyzer", "stub"]
    assert evaluate.main(["run", "--name", "t1", *common]) == 0
    run = tmp_path / "runs" / "t1"
    lines = (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["all"]["detection"]["videos"] == 2
    assert (run / "videos.csv").exists()
    assert evaluate.main(["run", "--name", "t1", *common]) == 0
    assert "0 videos to process, 2 cached" in capsys.readouterr().out
    assert evaluate.main(["score", "--run", str(run), "--labels", str(labels_path), "--tau", "0.0", "--holdout", "s02"]) == 0
    assert "holdout" in json.loads((run / "metrics.json").read_text(encoding="utf-8"))
