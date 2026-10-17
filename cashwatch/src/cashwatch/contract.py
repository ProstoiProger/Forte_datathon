import base64
import binascii
from typing import Any

TOP_KEYS = {"job_id", "status", "verdict", "violations"}
VIOLATION_KEYS = {"type", "timecode_sec", "description", "evidence_frames"}
FRAME_KEYS = {"timestamp_sec", "image_base64"}
STATUSES = {"processing", "done", "failed"}
VERDICTS = {"violation", "no_violation"}
JPEG_MAGIC = b"\xff\xd8\xff"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_frame(frame: Any, where: str, duration: float | None) -> list[str]:
    if not isinstance(frame, dict):
        return [f"{where}: not an object"]
    errors = []
    missing = FRAME_KEYS - frame.keys()
    if missing:
        errors.append(f"{where}: missing {sorted(missing)}")
        return errors
    ts = frame["timestamp_sec"]
    if not _is_number(ts) or ts < 0:
        errors.append(f"{where}.timestamp_sec: invalid {ts!r}")
    elif duration is not None and ts > duration + 0.5:
        errors.append(f"{where}.timestamp_sec: {ts} beyond duration {duration:.2f}")
    data = frame["image_base64"]
    if not isinstance(data, str) or not data:
        errors.append(f"{where}.image_base64: empty")
        return errors
    if data.startswith("data:"):
        errors.append(f"{where}.image_base64: must be raw base64 without data URI prefix")
        return errors
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        errors.append(f"{where}.image_base64: not valid base64")
        return errors
    if not raw.startswith(JPEG_MAGIC):
        errors.append(f"{where}.image_base64: not a JPEG")
    return errors


def validate_violation(item: Any, where: str, duration: float | None) -> list[str]:
    if not isinstance(item, dict):
        return [f"{where}: not an object"]
    errors = []
    missing = VIOLATION_KEYS - item.keys()
    if missing:
        return [f"{where}: missing {sorted(missing)}"]
    if not isinstance(item["type"], str) or not item["type"].strip():
        errors.append(f"{where}.type: empty")
    if not isinstance(item["description"], str) or not item["description"].strip():
        errors.append(f"{where}.description: empty")
    tc = item["timecode_sec"]
    if not _is_number(tc) or tc < 0:
        errors.append(f"{where}.timecode_sec: invalid {tc!r}")
    elif duration is not None and tc > duration + 0.5:
        errors.append(f"{where}.timecode_sec: {tc} beyond duration {duration:.2f}")
    frames = item["evidence_frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
        errors.append(f"{where}.evidence_frames: need 1-3 frames")
    else:
        for i, frame in enumerate(frames):
            errors.extend(validate_frame(frame, f"{where}.evidence_frames[{i}]", duration))
    return errors


def validate_result(payload: Any, duration: float | None = None, strict_keys: bool = True) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload: not an object"]
    errors = []
    if payload.get("status") != "done":
        return [f"status: expected 'done', got {payload.get('status')!r}"]
    missing = TOP_KEYS - payload.keys()
    if missing:
        return [f"payload: missing {sorted(missing)}"]
    if strict_keys and payload.keys() != TOP_KEYS:
        errors.append(f"payload: unexpected keys {sorted(payload.keys() - TOP_KEYS)}")
    if not isinstance(payload["job_id"], str) or not payload["job_id"]:
        errors.append("job_id: empty")
    if payload["verdict"] not in VERDICTS:
        errors.append(f"verdict: invalid {payload['verdict']!r}")
    violations = payload["violations"]
    if not isinstance(violations, list):
        return errors + ["violations: not a list"]
    if payload["verdict"] == "no_violation" and violations:
        errors.append("verdict no_violation with non-empty violations")
    if payload["verdict"] == "violation" and not violations:
        errors.append("verdict violation with empty violations")
    for i, item in enumerate(violations):
        errors.extend(validate_violation(item, f"violations[{i}]", duration))
        if strict_keys and isinstance(item, dict) and item.keys() != VIOLATION_KEYS:
            errors.append(f"violations[{i}]: unexpected keys {sorted(item.keys() - VIOLATION_KEYS)}")
        for j, frame in enumerate(item.get("evidence_frames", []) if isinstance(item, dict) else []):
            if strict_keys and isinstance(frame, dict) and frame.keys() != FRAME_KEYS:
                errors.append(f"violations[{i}].evidence_frames[{j}]: unexpected keys")
    return errors
