from pathlib import Path
from typing import Any

from cashwatch.analyzers.base import Event, EvidenceRef
from cashwatch.rules import Citation, Registry, ViolationType
from cashwatch.video import VideoInfo, draw_box, encode_jpeg_b64, frames_at

MAX_FRAMES = 3
CONTRACT_KEYS = ("type", "timecode_sec", "description", "evidence_frames")


def _clamp(t: float, info: VideoInfo) -> float:
    return round(min(max(t, 0.0), info.duration_sec), 2)


def _citation_dict(c: Citation) -> dict[str, Any]:
    return {
        "document": c.act.title,
        "approved_by": c.act.approved_by,
        "clause": c.clause,
        "quote": c.quote,
        "url": c.act.url,
    }


def describe(event: Event, vtype: ViolationType) -> str:
    parts = [event.observation.strip(), vtype.rationale.strip()]
    cited = vtype.verified_citations
    if cited:
        parts.append(f"Основание: {cited[0].text()}.")
    return " ".join(p for p in parts if p)


def render_violations(events: list[Event], video_path: Path, info: VideoInfo, registry: Registry) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda e: e.timecode_sec)
    refs = []
    for event in ordered:
        chosen = list(event.evidence[:MAX_FRAMES]) or [EvidenceRef(event.timecode_sec)]
        refs.append(chosen)
    flat_times = [_clamp(r.timestamp_sec, info) for chosen in refs for r in chosen]
    frames = frames_at(video_path, flat_times)
    if len(frames) != len(flat_times):
        raise RuntimeError("could not extract evidence frames")
    out, cursor = [], 0
    for event, chosen in zip(ordered, refs):
        vtype = registry.get(event.type_id)
        evidence = []
        for ref in chosen:
            ts, frame = frames[cursor]
            cursor += 1
            image = draw_box(frame, ref.box) if ref.box else frame
            evidence.append({"timestamp_sec": round(ts, 2), "image_base64": encode_jpeg_b64(image)})
        cited = vtype.verified_citations
        out.append(
            {
                "type": vtype.title,
                "timecode_sec": _clamp(event.timecode_sec, info),
                "description": describe(event, vtype),
                "evidence_frames": evidence,
                "type_id": vtype.type_id,
                "title": vtype.title,
                "confidence": round(float(event.confidence), 3),
                "observation": event.observation,
                "rule": _citation_dict(cited[0]) if cited else None,
                "rules": [_citation_dict(c) for c in cited],
            }
        )
    return out


def public_violation(item: dict[str, Any], detail: bool) -> dict[str, Any]:
    if detail:
        return dict(item)
    return {k: item[k] for k in CONTRACT_KEYS}
