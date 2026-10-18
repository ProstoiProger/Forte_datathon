from pathlib import Path
from typing import Any

from cashwatch.analyzers.base import Event, EvidenceRef
from cashwatch.rules import Citation, Registry, ViolationType
from cashwatch.video import VideoInfo, draw_box, encode_jpeg_b64, frames_at

MAX_FRAMES = 3
CONTRACT_KEYS = ("type", "timecode_sec", "description", "evidence_frames")
OTHER_TITLE = "Иное возможное нарушение кассовой дисциплины"


def _clamp(t: float, info: VideoInfo) -> float:
    return round(min(max(t, 0.0), info.duration_sec), 2)


def _citation_dict(c: Citation) -> dict[str, Any]:
    return {
        "document": c.act.title,
        "approved_by": c.act.approved_by,
        "clause": c.clause,
        "quote": c.quote,
        "url": c.act.url,
        "verified": True,
        "source": "registry",
    }


def _searched_norm_dict(norm: dict[str, str]) -> dict[str, Any]:
    """Same keys as a registry citation; `verified` is False because the clause was found by search."""
    return {
        "document": norm.get("title", ""),
        "approved_by": "",
        "clause": norm.get("citation", ""),
        "quote": norm.get("quote", ""),
        "url": norm.get("url", ""),
        "verified": False,
        "source": norm.get("source", "rag"),
    }


def describe(event: Event, vtype: ViolationType) -> str:
    parts = [event.observation.strip(), vtype.rationale.strip()]
    cited = vtype.verified_citations
    if cited:
        parts.append(f"Основание: {cited[0].text()}.")
    return " ".join(p for p in parts if p)


def describe_unlisted(event: Event) -> str:
    """A type outside the registry: the norm, if any, was found by search and is labeled as unverified."""
    parts = [event.observation.strip()]
    if event.norm:
        parts.append(f"Возможное основание (подобрано поиском по НПА, требует проверки): {event.norm['citation']}.")
    return " ".join(p for p in parts if p) or OTHER_TITLE


def _unlisted_fields(event: Event) -> dict[str, Any]:
    title = (event.title or "").strip() or OTHER_TITLE
    return {
        "type": title,
        "description": describe_unlisted(event),
        "type_id": event.type_id,
        "title": title,
        "rule": _searched_norm_dict(event.norm) if event.norm else None,
        "rules": [_searched_norm_dict(event.norm)] if event.norm else [],
    }


def _registry_fields(event: Event, vtype: ViolationType) -> dict[str, Any]:
    cited = vtype.verified_citations
    return {
        "type": vtype.title,
        "description": describe(event, vtype),
        "type_id": vtype.type_id,
        "title": vtype.title,
        "rule": _citation_dict(cited[0]) if cited else None,
        "rules": [_citation_dict(c) for c in cited],
    }


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
        evidence = []
        for ref in chosen:
            ts, frame = frames[cursor]
            cursor += 1
            image = draw_box(frame, ref.box) if ref.box else frame
            evidence.append({"timestamp_sec": round(ts, 2), "image_base64": encode_jpeg_b64(image)})
        fields = _registry_fields(event, registry.get(event.type_id)) if event.type_id in registry else _unlisted_fields(event)
        out.append(
            {
                "type": fields["type"],
                "timecode_sec": _clamp(event.timecode_sec, info),
                "description": fields["description"],
                "evidence_frames": evidence,
                "type_id": fields["type_id"],
                "title": fields["title"],
                "confidence": round(float(event.confidence), 3),
                "observation": event.observation,
                "rule": fields["rule"],
                "rules": fields["rules"],
            }
        )
    return out


def public_violation(item: dict[str, Any], detail: bool) -> dict[str, Any]:
    if detail:
        return dict(item)
    return {k: item[k] for k in CONTRACT_KEYS}
