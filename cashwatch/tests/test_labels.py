import importlib.util
from pathlib import Path

import openpyxl
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_labels", ROOT / "scripts" / "build_labels.py")
build_labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_labels)


@pytest.mark.parametrize("text,expected", [
    ("22 секунда", 22.0),
    ("5 секунда", 5.0),
    ("1минута 8 секунда", 68.0),
    ("2 минуты", 120.0),
    (None, None),
    ("", None),
])
def test_parse_seconds(text, expected) -> None:
    assert build_labels.parse_seconds(text) == expected


def test_canonical_id_maps_cyrillic_x() -> None:
    assert build_labels.canonical_id("Х-14") == "X-14"
    assert build_labels.canonical_id("X-1") == "X-1"


def test_groups_cover_known_ids() -> None:
    assert build_labels.group_of(47) == build_labels.group_of(4) == "s01"
    assert build_labels.group_of(99) == "x99"


def _sheet(path: Path, rows: list[tuple]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(("наименование файла", "таймстемп", "нарушение / без нарушения"))
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def test_build_merges_duplicate_files(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "Х-4.mov").write_bytes(b"same")
    (videos / "Х-47.mov").write_bytes(b"same")
    (videos / "Х-5.mov").write_bytes(b"clean")
    (videos / "Х-48.mov").write_bytes(b"clean")
    xlsx = _sheet(tmp_path / "l.xlsx", [
        ("Х-4", "57 секунда", "нарушение"),
        ("Х-5", None, "без нарушения"),
        ("Х-47", "1минута 8 секунда", "нарушение"),
        ("Х-48", None, "без нарушения"),
    ])
    labels = build_labels.build(xlsx, videos)
    by_id = {v["video"]: v for v in labels["videos"]}
    assert sorted(by_id) == ["X-4", "X-5"]
    assert [e["t"] for e in by_id["X-4"]["events"]] == [57.0, 68.0]
    assert by_id["X-4"]["duplicates"] == ["X-47"]
    assert by_id["X-4"]["events"][0]["type"] == "unauthorized_entry"
    assert by_id["X-5"]["label"] == "no_violation" and by_id["X-5"]["duplicates"] == ["X-48"]


def test_build_rejects_conflicting_duplicate(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "X-1.mov").write_bytes(b"same")
    (videos / "X-2.mov").write_bytes(b"same")
    xlsx = _sheet(tmp_path / "l.xlsx", [("X-1", "3 секунда", "нарушение"), ("X-2", None, "без нарушения")])
    with pytest.raises(ValueError):
        build_labels.build(xlsx, videos)


def test_build_requires_video_files(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    xlsx = _sheet(tmp_path / "l.xlsx", [("X-9", None, "без нарушения")])
    with pytest.raises(FileNotFoundError):
        build_labels.build(xlsx, videos)
