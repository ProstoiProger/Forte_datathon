"""Build the empirical case memory from the labeled training videos.

Stages (each writes into CASES_DIR and can be rerun on its own):
  labels    spreadsheet -> labels.json (40 unique videos, events, groups, duplicates) + media probe
  frames    decode every video once at FRAME_FPS -> frames/<video>/; clip windows -> windows.json;
            10 s review clips around each labeled event -> clips/
  annotate  VLM passes (facts, events, overview) -> vlm/*.jsonl
  cases     merge labels, VLM output, manual review and the rule registry -> cases.jsonl + cards.jsonl
  embed     Qwen3-VL-Embedding vectors -> vectors/{clip,frame,card}.npy + .jsonl
"""

import json
import logging
from pathlib import Path

from .. import config, media
from . import annotate as ann
from .labels import build_labels
from .windows import NEAR_SEC, build_windows, relation

log = logging.getLogger(__name__)

REVIEW_FILE = config.PROJECT / "cases_review.json"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.labels = root / "labels.json"
        self.frames = root / "frames"
        self.frames_index = root / "frames.json"
        self.windows = root / "windows.json"
        self.clips = root / "clips"
        self.vlm = root / "vlm"
        self.cases = root / "cases.jsonl"
        self.cards = root / "cards.jsonl"
        self.vectors = root / "vectors"
        self.review_template = root / "review_template.json"


def load_types(registry: Path) -> dict:
    """Violation types with verified citations from the cashwatch rule registry (the placeholder type is dropped)."""
    if not registry.exists():
        log.warning("rule registry %s not found: cases get no candidate rules", registry)
        return {}
    types = _read(registry)["types"]
    return {k: v for k, v in types.items() if k != "stub_motion"}


def rule_ids(type_entry: dict | None) -> list[dict]:
    if not type_entry:
        return []
    return [{"act": c["act"], "clause": c["clause"], "quote": c["quote"]}
            for c in type_entry.get("citations", []) if c.get("verified")]


# ---------------------------------------------------------------- stages


def stage_labels(p: Paths) -> dict:
    labels = build_labels(config.LABELS_XLSX, config.VIDEOS_DIR)
    for v in labels["videos"]:
        v["media"] = media.probe(config.VIDEOS_DIR / v["file"])
    _write(p.labels, labels)
    vids = labels["videos"]
    pos = sum(v["label"] == "violation" for v in vids)
    log.info("videos=%d violation=%d clean=%d events=%d groups=%d", len(vids), pos, len(vids) - pos,
             sum(len(v["events"]) for v in vids), len({v["group"] for v in vids}))
    for v in vids:
        if v["duplicates"]:
            log.info("merged %s <- %s", v["video"], ", ".join(v["duplicates"]))
    return labels


def stage_frames(p: Paths, force: bool = False) -> None:
    labels = _read(p.labels)
    index = _read(p.frames_index) if p.frames_index.exists() and not force else {}
    windows = []
    for v in labels["videos"]:
        src = config.VIDEOS_DIR / v["file"]
        dur = v["media"]["duration"]
        if v["video"] not in index:
            index[v["video"]] = media.decode_frames(src, p.frames / v["video"], config.FRAME_FPS, dur)
            _write(p.frames_index, index)
            log.info("frames %s: %d", v["video"], len(index[v["video"]]))
        windows += build_windows(v, index[v["video"]], dur, config.WINDOW_SEC, config.WINDOW_STRIDE_SEC)
        for e in v["events"]:
            start, end = max(0.0, e["t"] - config.CLIP_HALF_SEC), min(dur, e["t"] + config.CLIP_HALF_SEC)
            out = p.clips / f"{e['event_id'].replace('#', '_')}.mp4"
            if force or not out.exists():
                media.export_clip(src, out, start, end)
    _write(p.windows, windows)
    rel = {}
    for w in windows:
        rel[w["relation"]] = rel.get(w["relation"], 0) + 1
    log.info("windows=%d %s", len(windows), rel)


def stage_annotate(p: Paths, passes: list[str], videos: list[str] | None = None) -> None:
    from ..vlm import Ollama
    labels = _read(p.labels)
    index = _read(p.frames_index)
    vlm = Ollama(config.OLLAMA_URL, config.VLM_MODEL, config.VLM_NUM_CTX, config.VLM_TIMEOUT_SEC)
    a = ann.Annotator(vlm, p.frames, p.vlm, load_types(config.CASHWATCH_REGISTRY))
    selected = [v for v in labels["videos"] if not videos or v["video"] in videos]
    if "events" in passes:
        a.events([j for v in selected for j in ann.event_jobs(v, index[v["video"]], v["media"]["duration"])])
    if "overview" in passes:
        a.overview([ann.overview_job(v, index[v["video"]], v["media"]["duration"]) for v in selected])
    if "facts" in passes:
        a.facts([j for v in selected for j in ann.facts_jobs(v, index[v["video"]], v["media"]["duration"])])


# ---------------------------------------------------------------- case assembly


def _timeline(facts: dict[str, dict], video: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Per-frame states, actions and per-window scene lines of one video from the facts pass."""
    states, actions, scenes = [], [], []
    for rec in sorted((r for r in facts.values() if r["video"] == video), key=lambda r: r["start"]):
        times = rec["frame_times"]
        parsed = rec["parsed"]
        for fr in parsed.get("frames", []):
            i = fr.get("frame", 0) - 1
            if 0 <= i < len(times):
                states.append({"t": times[i], **{k: fr.get(k) for k in ann.CONCEPTS}})
        for act in parsed.get("actions", []):
            a, b = act.get("from_frame", 1) - 1, act.get("to_frame", 1) - 1
            a, b = min(max(a, 0), len(times) - 1), min(max(b, 0), len(times) - 1)
            actions.append({"t_start": times[min(a, b)], "t_end": times[max(a, b)], "actor": act.get("actor"),
                            "action": act.get("action"), "object": act.get("object")})
        scenes.append({"start": rec["start"], "end": rec["end"], "scene_ru": parsed.get("scene_ru", ""),
                       "states": parsed.get("frames", [])})
    states.sort(key=lambda s: s["t"])
    return states, actions, scenes


def state_changes(states: list[dict], start: float, end: float) -> list[dict]:
    """Concept flips between consecutive annotated frames inside [start, end]."""
    sel = [s for s in states if start <= s["t"] <= end]
    out = []
    for prev, cur in zip(sel, sel[1:]):
        for k in ann.CONCEPTS:
            if prev.get(k) is not None and cur.get(k) is not None and prev[k] != cur[k]:
                out.append({"t": cur["t"], "concept": k, "from": prev[k], "to": cur[k]})
    return out


def _review_template(labels: dict, events_vlm: dict, types: dict) -> dict:
    out = {}
    for v in labels["videos"]:
        for e in v["events"]:
            vlm = (events_vlm.get(e["event_id"]) or {}).get("parsed") or {}
            out[e["event_id"]] = {
                "type": e["type"],
                "type_status": e["type_status"],
                "vlm_type": vlm.get("violation_type"),
                "title": (types.get(e["type"]) or {}).get("title"),
                "notes": "",
            }
    return {"note": "copy to knowledge/cases_review.json, watch clips/<event>.mp4, set type and "
                    "type_status=confirmed; new types: give type=new and a title", "events": out}


def _frame_review(path: Path) -> dict[str, list[dict]]:
    """Hand-labelled frames of the VLM bake-off (Datathon/runs/vlm_bakeoff/gt.json), keyed by video."""
    from .labels import canonical_id
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for key, facts in _read(path)["frames"].items():
        name, t = key.split("@")
        out.setdefault(canonical_id(name), []).append({"t": float(t), "facts": facts,
                                                       "source": "frame review 2026-10-17"})
    return out


def stage_cases(p: Paths) -> None:
    labels = _read(p.labels)
    index = _read(p.frames_index)
    types = load_types(config.CASHWATCH_REGISTRY)
    facts = ann.load_pass(p.vlm, "facts")
    events_vlm = ann.load_pass(p.vlm, "events")
    overview = ann.load_pass(p.vlm, "overview")
    review = _read(REVIEW_FILE).get("events", {}) if REVIEW_FILE.exists() else {}
    manual = _frame_review(config.FRAME_REVIEW_JSON)
    _write(p.review_template, _review_template(labels, events_vlm, types))

    by_group: dict[str, list[dict]] = {}
    for v in labels["videos"]:
        by_group.setdefault(v["group"], []).append(v)

    cases, cards = [], []
    for v in labels["videos"]:
        vid, dur = v["video"], v["media"]["duration"]
        frames = index[vid]
        states, actions, scenes = _timeline(facts, vid)
        ov = (overview.get(vid) or {}).get("parsed")
        events = []
        for e in v["events"]:
            rv = review.get(e["event_id"], {})
            reviewed = rv.get("type_status") == "confirmed"
            type_id = rv.get("type") or e["type"]
            status = rv.get("type_status") or e["type_status"]
            # registry types keep their registry title; a reviewer's title is used only for a type outside it
            title = (types.get(type_id) or {}).get("title") or rv.get("title")
            vlm = (events_vlm.get(e["event_id"]) or {})
            parsed = vlm.get("parsed") or {}
            ftimes = vlm.get("frame_times") or []
            ev_frame = parsed.get("evidence_frame")
            keyframes = [f for f in frames if abs(f["t"] - e["t"]) <= 2.0 + 1e-6]
            events.append({
                "event_id": e["event_id"],
                "t": e["t"],
                "clip": {"start": max(0.0, e["t"] - config.CLIP_HALF_SEC), "end": min(dur, e["t"] + config.CLIP_HALF_SEC),
                         "file": f"clips/{e['event_id'].replace('#', '_')}.mp4"},
                "keyframes": [{"t": f["t"], "file": f"frames/{vid}/{f['file']}"} for f in keyframes],
                "state_changes": state_changes(states, e["t"] - 4, e["t"] + 4),
                "actions": [a for a in actions if a["t_end"] >= e["t"] - 4 and a["t_start"] <= e["t"] + 4],
                "vlm": {**{k: parsed.get(k) for k in ("before_ru", "at_label_ru", "after_ru", "visual_events",
                                                      "violation_type", "new_type_title_ru", "distinguishing_cue_ru",
                                                      "confidence")},
                        "evidence_t": ftimes[ev_frame - 1] if isinstance(ev_frame, int) and 0 < ev_frame <= len(ftimes) else None,
                        "model": vlm.get("model"), "provenance": "label-conditioned VLM, unreviewed"} if parsed else None,
                "type": {"id": type_id, "title": title,
                         "status": status, "source": "manual review" if reviewed else "frame-review hypothesis",
                         "vlm_agrees": (parsed.get("violation_type") == type_id) if parsed else None},
                "candidate_rules": rule_ids(types.get(type_id)),
                # while the type is unconfirmed and the VLM sees another type, keep that type's rules for the reviewer
                "alt_type": ({"id": parsed["violation_type"],
                              "title": (types.get(parsed["violation_type"]) or {}).get("title")
                              or parsed.get("new_type_title_ru") or None,
                              "candidate_rules": rule_ids(types.get(parsed["violation_type"]))}
                             if parsed and status != "confirmed" and parsed.get("violation_type") not in (type_id, "not_visible")
                             else None),
                "review": {"status": "reviewed" if reviewed else "unreviewed",
                           "notes": rv.get("notes", "")},
            })
        same_setup = [o["video"] for o in by_group[v["group"]] if o["video"] != vid]
        case = {
            "video_id": vid, "file": v["file"], "md5": v["md5"], "duplicates": v["duplicates"], "group": v["group"],
            "media": v["media"],
            "ground_truth": {"label": v["label"], "events": [{"event_id": e["event_id"], "t": e["t"]} for e in v["events"]]},
            "same_setup_videos": same_setup,
            "overview": ov,
            "events": events,
            "timeline": states,
            "actions": actions,
            "scenes": scenes,
            "manual_frame_facts": manual.get(vid, []),
        }
        cases.append(case)
        cards += _cards(case, v)

    with p.cases.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with p.cards.open("w", encoding="utf-8") as fh:
        for c in cards:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_ev = sum(len(c["events"]) for c in cases)
    log.info("cases=%d events=%d (vlm-described %d) cards=%d timeline-frames=%d", len(cases), n_ev,
             sum(1 for c in cases for e in c["events"] if e["vlm"]), len(cards), sum(len(c["timeline"]) for c in cases))


STATE_RU = {
    "cashier_at_workplace": "кассир на рабочем месте", "cashier_leaving": "кассир уходит с рабочего места",
    "cash_on_desk": "наличные на столе", "cash_in_hands": "наличные в руках кассира", "safe_open": "сейф открыт",
    "door_open": "дверь открыта", "phone_in_hand": "телефон в руке кассира", "drinking": "кассир пьёт или держит напиток",
    "personal_items_on_desk": "личные вещи на столе", "collection_bag": "инкассаторская сумка",
    "visitor_present": "в помещении есть посетитель",
}


def _states_ru(frames: list[dict]) -> str:
    on = [k for k in STATE_RU if any(f.get(k) for f in frames)]
    people = max((f.get("people") or 0 for f in frames), default=0)
    return f"людей в помещении: {people}; " + ("; ".join(STATE_RU[k] for k in on) if on else "особых состояний нет")


def _cards(case: dict, video: dict) -> list[dict]:
    """Russian text cards for text retrieval: one per event, one per video, one per annotated window."""
    vid, group, label = case["video_id"], case["group"], case["ground_truth"]["label"]
    base = {"video": vid, "group": group, "label": label}
    type_of = {e["event_id"]: e["type"]["id"] for e in case["events"]}  # reviewed type where available
    cards = []
    for e in case["events"]:
        vlm = e["vlm"] or {}
        rules = "; ".join(f"{r['act']} {r['clause']}" for r in e["candidate_rules"])
        type_ru = e["type"]["title"] or e["type"]["id"] or "не определён"
        if e["type"]["status"] != "confirmed":
            type_ru = f"гипотеза, не подтверждена: {type_ru}"
        if e["alt_type"]:
            type_ru += f"; по описанию модели: {e['alt_type']['title'] or e['alt_type']['id']}"
        text = (f"Возможное нарушение кассовой дисциплины в {e['t']:.0f} с. "
                f"До: {vlm.get('before_ru') or '-'} В момент: {vlm.get('at_label_ru') or '-'} "
                f"После: {vlm.get('after_ru') or '-'} Отличие от нормы: {vlm.get('distinguishing_cue_ru') or '-'} "
                f"Тип ({type_ru}). Нормы: {rules or 'не сопоставлены'}.")
        cards.append({"id": f"card:{e['event_id']}", "kind": "event", **base, "event_id": e["event_id"],
                      "t": e["t"], "type": e["type"]["id"], "relation": "event", "text": text})
    ov = case["overview"]
    if ov:
        text = (f"{'Видео с нарушением' if label == 'violation' else 'Видео без нарушений (нормальная работа)'}. "
                f"Планировка: {ov.get('layout_ru')} Люди: {ov.get('people_ru')} Ход событий: {ov.get('procedure_ru')}")
        cards.append({"id": f"card:{vid}", "kind": "video", **base, "event_id": None, "t": None, "type": None,
                      "relation": "video", "text": text})
    for sc in case["scenes"]:
        rel, ev = relation(sc["start"], sc["end"], video["events"], label == "violation")
        text = f"{sc['scene_ru']} Состояние: {_states_ru(sc['states'])}."
        cards.append({"id": f"card:{vid}@{sc['start']:06.2f}", "kind": "window", **base,
                      "event_id": ev["event_id"] if ev else None, "t": sc["start"], "end": sc["end"],
                      "type": type_of.get(ev["event_id"]) if ev and rel == "event" else None,
                      "relation": rel, "text": text})
    return cards


# ---------------------------------------------------------------- embeddings


def _point_relation(t: float, video: dict) -> str:
    if video["label"] != "violation":
        return "normal"
    d = min(abs(t - e["t"]) for e in video["events"])
    return "event" if d <= 2.0 else "near" if d <= 2.0 + NEAR_SEC else "other"


def stage_embed(p: Paths, spaces: list[str], batch_size: int = 8) -> None:
    from ..embedder import CARD, CLIP, Embedder, Item
    from ..store import Space
    from ..vlm import Ollama

    try:
        freed = Ollama(config.OLLAMA_URL, config.VLM_MODEL).unload_all()
        if freed:
            log.info("unloaded from Ollama to free VRAM: %s", ", ".join(freed))
    except Exception as exc:  # Ollama not running is fine here
        log.info("ollama not reachable (%s); continuing", exc)

    labels = {v["video"]: v for v in _read(p.labels)["videos"]}
    emb = Embedder(config.EMBED_MODEL, config.EMBED_DIM)
    try:
        if "clip" in spaces:
            windows = _read(p.windows)
            items = [Item(frames=[p.frames / w["video"] / f for f in w["frames"]], instruction=CLIP) for w in windows]
            vecs = emb.encode(items, batch_size=batch_size, progress="clip")
            meta = [{k: w[k] for k in w if k != "frames"} for w in windows]
            Space("clip", vecs, meta).save(p.vectors)
        if "frame" in spaces:
            index = _read(p.frames_index)
            meta, items = [], []
            for vid, frames in index.items():
                for f in frames:
                    meta.append({"id": f"{vid}@t{f['t']:07.2f}", "video": vid, "group": labels[vid]["group"],
                                 "label": labels[vid]["label"], "t": f["t"], "file": f"frames/{vid}/{f['file']}",
                                 "relation": _point_relation(f["t"], labels[vid])})
                    items.append(Item(image=p.frames / vid / f["file"], instruction=CLIP))
            vecs = emb.encode(items, batch_size=max(batch_size, 16), progress="frame")
            Space("frame", vecs, meta).save(p.vectors)
        if "card" in spaces:
            if not p.cards.exists():
                raise FileNotFoundError("cards.jsonl missing: run the cases stage first")
            cards = _jsonl(p.cards)
            vecs = emb.encode([Item(text=c["text"], instruction=CARD) for c in cards], batch_size=32, progress="card")
            Space("card", vecs, cards).save(p.vectors)
    finally:
        emb.unload()
