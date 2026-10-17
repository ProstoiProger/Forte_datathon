"""Organizers' spreadsheet -> one record per unique video.

Same rules as cashwatch/scripts/build_labels.py, so both projects agree on ids, groups and duplicates:
files with identical content are merged into one video with several events (X-4 = X-47, X-5 = X-48),
every video gets a camera-setup group, every event a type hypothesis from frame review.
Type hypotheses are not ground truth: the organizers give only violation / no violation and the second.
"""

import hashlib
import re
import unicodedata
from pathlib import Path

import openpyxl

# Camera setups (same room layout and actors), used for leave-one-setup-out evaluation.
GROUPS = [
    [1, 2, 3, 4, 5, 47, 48],
    [8],
    [10],
    [11, 12, 14],
    [13, 15],
    [16, 17],
    [18, 19],
    [20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33],
    [34, 35, 37, 38, 39],
    [40, 43, 44, 45, 46],
]

# (video number, labeled second) -> (type id from the cashwatch registry, confidence of the hypothesis)
TYPE_HYPOTHESES = {
    (1, 22): ("cash_handling", "uncertain"),
    (2, 10): ("phone_use", "likely"),
    (4, 57): ("unauthorized_entry", "likely"),
    (47, 68): ("cash_handling", "uncertain"),
    (11, 18): ("workplace_abandoned", "likely"),
    (13, 47): ("safe_left_open", "uncertain"),
    (15, 15): ("workplace_abandoned", "likely"),
    (17, 7): ("collector_handover", "likely"),
    (18, 20): ("unauthorized_entry", "likely"),
    (20, 5): ("safe_left_open", "likely"),
    (22, 14): ("workplace_abandoned", "likely"),
    (24, 8): ("workplace_abandoned", "likely"),
    (26, 9): ("cash_unsecured", "uncertain"),
    (28, 7): ("workplace_abandoned", "likely"),
    (30, 13): ("unauthorized_entry", "likely"),
    (32, 5): ("door_left_open", "uncertain"),
    (33, 6): ("door_left_open", "uncertain"),
    (34, 13): ("unauthorized_entry", "uncertain"),
    (38, 33): ("unauthorized_entry", "likely"),
    (40, 17): ("personal_items", "likely"),
    (44, 16): ("phone_use", "uncertain"),
    (46, 11): ("food_drink", "likely"),
}

VIOLATION = "нарушение"


def canonical_id(name: str) -> str:
    """'Х-24' with a Cyrillic Х and 'X-24' with a Latin X are the same video."""
    return unicodedata.normalize("NFC", name).replace("Х", "X").replace("х", "x").strip()


def number_of(video_id: str) -> int:
    return int(re.findall(r"\d+", video_id)[0])


def parse_seconds(text) -> float | None:
    """'57 секунд' -> 57, '1минута 8 секунд' -> 68, empty -> None."""
    if text is None or not str(text).strip():
        return None
    s = str(text).lower()
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        raise ValueError(f"cannot parse timestamp {text!r}")
    if "мин" in s:
        return float(nums[0] * 60 + (nums[1] if len(nums) > 1 else 0))
    return float(nums[0])


def group_of(num: int) -> str:
    for i, members in enumerate(GROUPS, 1):
        if num in members:
            return f"s{i:02d}"
    return f"x{num:02d}"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(xlsx: Path) -> list[dict]:
    sheet = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    rows = []
    for name, stamp, label in sheet.iter_rows(min_row=2, max_col=3, values_only=True):
        if not name:
            continue
        rows.append({"name": str(name).strip(), "t": parse_seconds(stamp),
                     "violation": str(label).strip().lower() == VIOLATION})
    return rows


def build_labels(xlsx: Path, videos_dir: Path) -> dict:
    files = {canonical_id(p.stem): p for p in videos_dir.iterdir() if p.suffix.lower() in (".mov", ".mp4")}
    by_hash: dict[str, dict] = {}
    order: list[str] = []
    for row in read_rows(xlsx):
        vid = canonical_id(row["name"])
        if vid not in files:
            raise FileNotFoundError(f"no video file for label {row['name']!r}")
        path = files[vid]
        digest = md5(path)
        if digest not in by_hash:
            by_hash[digest] = {
                "video": vid,
                "file": path.name,
                "md5": digest,
                "group": group_of(number_of(vid)),
                "label": "no_violation",
                "events": [],
                "duplicates": [],
            }
            order.append(digest)
        entry = by_hash[digest]
        if vid != entry["video"]:
            entry["duplicates"].append(vid)
        if row["violation"]:
            if row["t"] is None:
                raise ValueError(f"{vid}: violation without timestamp")
            type_id, status = TYPE_HYPOTHESES.get((number_of(vid), int(row["t"])), (None, "unknown"))
            entry["label"] = "violation"
            entry["events"].append({"t": row["t"], "type": type_id, "type_status": status, "source": vid})
        elif entry["events"]:
            raise ValueError(f"{vid}: duplicate of {entry['video']} labeled differently")
    videos = [by_hash[d] for d in order]
    for v in videos:
        v["events"].sort(key=lambda e: e["t"])
        for i, e in enumerate(v["events"], 1):
            e["event_id"] = f"{v['video']}#e{i}"
    return {
        "source": xlsx.name,
        "note": "type values are hypotheses from frame review; confirm by watching each event",
        "videos": videos,
    }
