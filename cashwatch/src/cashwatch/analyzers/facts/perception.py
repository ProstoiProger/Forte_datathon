"""VLM input/output of the facts analyzer: per-window facts (perception) and per-candidate verification.

The model is used only for what it perceives reliably: boolean states of a short window, and a yes/no check of ONE
norm against a few candidate frames next to reference examples. Deciding which windows are suspicious is left to
the case base.
"""

import base64
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from cashwatch.analyzers.facts.schema import FLAG_DESCRIPTIONS, FLAG_KEYS, PEOPLE_DESCRIPTION, CaseRef, FactWindow, Norm, Verification
from cashwatch.analyzers.vlm import Frame, VlmCallError

log = logging.getLogger("cashwatch.facts.perception")

MOTION_MAX_SIDE = 212  # an 848-px portrait frame becomes about 119x212
MOTION_PIXEL_DELTA = 25
MAX_QUOTE_CHARS = 1500
MAX_DESCRIPTION_CHARS = 400
DEADLINE_MARGIN_SEC = 2.0  # an attempt must end this long before the deadline
MIN_ATTEMPT_SEC = 3.0  # no attempt is started with less time than this

# Model text goes to the public API, which must not carry legal references (the prompt contains norm citations).
LEGAL_REFERENCE = re.compile(
    r"\bп{1,2}\.\s*\d|\b(?:ст|гл|ч)\.\s*\d|\bпункт|\bподпункт|\bстать|\bглав(?!н)|\bпараграф|\bпостановлен|\bинструкц|\bприказ"
    r"|\bправил(?!ьн)|\bзакон|\bкодекс|№|\bнбрк",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
ABBREVIATION_END = re.compile(r"(?:^|[\s(])(?:п|пп|ст|гл|ч|см|т\.\s?[едп])\.$", re.IGNORECASE)

FACTS_SYSTEM = (
    "You annotate footage from a fixed top-down CCTV camera in a bank cash room. "
    "Report only what is visible in the frames. "
    "Refer to people only by role (cashier, client, collector, other person); never describe or identify faces."
)

FACTS_PROMPT = """These {n} frames are in time order from one fixed top-down camera in a bank cash room:
{frames}

For the whole interval set each key to true if the state holds in ANY of the frames, otherwise false:
{flags}
Also give:
- people: {people}
- change_frame: number of the frame (1-{n}) where the scene changes most, 0 if nothing changes
Answer only with JSON."""

VERIFY_SYSTEM = (
    "You check frames from a fixed top-down CCTV camera in a bank cash room against cash-handling rules. "
    "A cashier works at a desk with a cash counting machine, a computer and a metal safe. Clients and cash collectors "
    "may enter through the room door. The candidate moment was selected automatically and is often normal work. "
    "Judge only what is directly visible in the candidate frames. Do not guess about documents, amounts, accounts "
    "or anything outside the frames. Refer to people only by role (cashier, client, collector, other person); "
    "never describe or identify faces."
)

REFERENCES_BLOCK = """Images 1-{r} are references from OTHER recordings. They only show what such situations look like; do not judge them and do not describe them in your answer:
{references}

Images {first}-{last} are the candidate frames to judge, in time order:
{frames}"""

FRAMES_BLOCK = """The candidate frames to judge ({n}), in time order:
{frames}"""

REGISTRY_QUESTION = """Question: do the candidate frames show the violation "{type_title}" under the norm{s} above?"""

SEARCH_QUESTION = """Question: the candidate frames show an unusual situation. The norms above are candidates: verified rule-registry norms first, then possibly clauses found by a text search, and most of them will not apply. Is any of these norms clearly violated by what is visible in the candidate frames? If so, which one?"""

VERIFY_PROMPT = """{images}

{norms}

{question}

How to decide:
- Normal work is not a violation: serving a client at the desk, counting cash in front of the client, putting cash into the safe and locking it, the cashier leaving after putting everything away with nothing left on the desk. Judge the visible signs given in the question, not the wording of the norm alone.
- Answer confirmed=true only if the violation is visible in the candidate frames. If it is unclear or not visible, answer confirmed=false.
- Do not invent documents, amounts or anything that is not in the frames.
- Frame times are seconds from the start of the clip; the last frame may be the end of the clip — use it to judge whether a state persists (for example a door or safe left open, or the cashier not returning).
- Do not cite or mention norms, clauses or documents in description_ru or title_ru.

Answer only with JSON:
- description_ru: one or two short sentences in Russian with only the visible facts that matter for the question; people only by role (кассир, клиент, инкассатор), do not describe faces;
- confirmed: true or false;
- frame: number of the first candidate frame (1-{n}) in which the violation is visible, i.e. where it starts, 0 if not confirmed;
- norm_index: {norm_hint};
- title_ru: {title_hint};
- confidence: probability from 0 to 1 that the candidate frames show the violation."""

REFERENCE_LABELS = {"violation": "example of a confirmed violation", "normal": "example of normal work"}


def _decode_gray(image_b64: str) -> np.ndarray | None:
    try:
        buf = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None


def window_motion(images_b64: list[str]) -> float:
    """Largest share of changed pixels between consecutive frames (0..1); 0 for fewer than 2 frames."""
    grays = [g for g in map(_decode_gray, images_b64) if g is not None]
    if len(grays) < 2:
        return 0.0
    h, w = grays[0].shape
    scale = min(1.0, MOTION_MAX_SIDE / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = [cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.int16) for g in grays]
    return max(float(np.mean(np.abs(a - b) > MOTION_PIXEL_DELTA)) for a, b in zip(small, small[1:]))


def facts_schema(n_frames: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            **{k: {"type": "boolean"} for k in FLAG_KEYS},
            "people": {"type": "integer", "minimum": 0},
            "change_frame": {"type": "integer", "minimum": 0, "maximum": n_frames},
        },
        "required": [*FLAG_KEYS, "people", "change_frame"],
    }


def verify_schema(n_frames: int, n_norms: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "description_ru": {"type": "string"},
            "confirmed": {"type": "boolean"},
            "frame": {"type": "integer", "minimum": 0, "maximum": n_frames},
            "norm_index": {"type": "integer", "minimum": 0, "maximum": n_norms},
            "title_ru": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["description_ru", "confirmed", "frame", "norm_index", "title_ru", "confidence"],
    }


def _frame_lines(frames: list[Frame], offset: int = 0) -> str:
    if offset:
        return "\n".join(f"Frame {i} (image {offset + i}): t={f.t:.1f}s" for i, f in enumerate(frames, 1))
    return "\n".join(f"Frame {i}: t={f.t:.1f}s" for i, f in enumerate(frames, 1))


def build_facts_prompt(frames: list[Frame]) -> str:
    flags = "\n".join(f"- {k}: {d}" for k, d in FLAG_DESCRIPTIONS.items())
    return FACTS_PROMPT.format(n=len(frames), frames=_frame_lines(frames), flags=flags, people=PEOPLE_DESCRIPTION)


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_QUOTE_CHARS else text[: MAX_QUOTE_CHARS - 1] + "…"


def _norm_lines(norms: list[Norm]) -> str:
    if not norms:
        return "Norm: no text available."
    return "\n\n".join(f'Norm {i}: {n.title}\n{n.citation}\n"{_quote(n.quote)}"' for i, n in enumerate(norms, 1))


def build_verify_prompt(frames: list[Frame], norms: list[Norm], refs: list[CaseRef], type_title: str | None) -> str:
    if refs:
        ref_lines = "\n".join(
            f"Image {i} - reference: {REFERENCE_LABELS.get(r.label, r.label)}. {r.description}" for i, r in enumerate(refs, 1)
        )
        offset = len(refs)
        images = REFERENCES_BLOCK.format(
            r=offset, references=ref_lines, first=offset + 1, last=offset + len(frames), frames=_frame_lines(frames, offset)
        )
    else:
        images = FRAMES_BLOCK.format(n=len(frames), frames=_frame_lines(frames))
    m = len(norms)
    if type_title:
        question = REGISTRY_QUESTION.format(type_title=type_title, s="s" if m > 1 else "")
        norm_hint = f"number of the violated norm (1-{m}), 0 if not confirmed" if m > 1 else ("1 if confirmed, otherwise 0" if m else "0")
        title_hint = "empty string"
    else:
        question = SEARCH_QUESTION
        norm_hint = f"number of the violated norm (1-{m}), 0 if none"
        title_hint = "short title of the violation in Russian (3-8 words), empty string if not confirmed"
    return VERIFY_PROMPT.format(
        images=images, norms=_norm_lines(norms), question=question, n=len(frames), norm_hint=norm_hint, title_hint=title_hint
    )


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    for piece in SENTENCE_END.split(text.strip()):
        if parts and ABBREVIATION_END.search(parts[-1]):
            parts[-1] = f"{parts[-1]} {piece}"
        elif piece:
            parts.append(piece)
    return parts


def scrub_legal(text: str) -> str:
    """Drops every sentence that refers to a legal act, clause or document number."""
    kept = [s for s in _sentences(text) if not LEGAL_REFERENCE.search(s)]
    return " ".join(kept)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _as_int(value: Any, default: int, low: int, high: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    number = max(low, number)
    return number if high is None else min(high, number)


def _as_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else max(low, min(high, number))


def _load_object(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def parse_facts(content: str, n_frames: int) -> tuple[dict[str, bool], int, int] | None:
    """(flags, people, change_frame), or None if the answer is not a JSON object with any of the expected keys."""
    data = _load_object(content)
    if data is None or not any(k in data for k in (*FLAG_KEYS, "people", "change_frame")):
        return None
    flags = {k: bool(_as_bool(data.get(k))) for k in FLAG_KEYS}
    return flags, _as_int(data.get("people"), 0, 0), _as_int(data.get("change_frame"), 0, 0, n_frames)


def parse_verification(content: str, n_frames: int, n_norms: int, registry: bool) -> Verification | None:
    data = _load_object(content)
    confirmed = _as_bool(data.get("confirmed")) if data is not None else None
    if confirmed is None:
        return None
    frame = _as_int(data.get("frame"), 0, 0)
    norm_index = _as_int(data.get("norm_index"), 0, 0)
    if not confirmed:
        frame, norm_index = 0, 0
    elif not 1 <= frame <= n_frames:
        frame = (n_frames + 1) // 2
    title = "" if registry else scrub_legal(str(data.get("title_ru") or ""))[:120]
    return Verification(
        confirmed=confirmed,
        frame=frame,
        description=scrub_legal(str(data.get("description_ru") or ""))[:MAX_DESCRIPTION_CHARS],
        confidence=_as_float(data.get("confidence"), 0.0, 0.0, 1.0),
        title=title or None,
        norm_index=norm_index if 1 <= norm_index <= n_norms else None,
        extra={"raw": data},
    )


class Perceiver:
    """All VLM calls of the facts analyzer. `client` is anything with chat(system, user, images, schema) -> str."""

    def __init__(self, client: Any, retries: int = 1) -> None:
        self.client = client
        self.retries = retries
        self.stats: dict[str, float] = {"calls": 0, "failed": 0, "skipped": 0, "seconds": 0.0}

    def _timeout(self, deadline: float | None) -> float | None:
        """Timeout for the next attempt, None for the client default, 0 if no attempt fits before the deadline."""
        if deadline is None:
            return None
        left = deadline - time.monotonic() - DEADLINE_MARGIN_SEC
        default = getattr(self.client, "timeout", None)
        timeout = min(default, left) if isinstance(default, (int, float)) else left
        return timeout if timeout >= MIN_ATTEMPT_SEC else 0.0

    def _ask(
        self, kind: str, system: str, user: str, images: list[str], schema: dict[str, Any], parse: Callable[[str], Any], deadline: float | None
    ) -> Any:
        started = time.monotonic()
        self.stats["calls"] += 1
        attempts = 0
        try:
            for attempt in range(1, self.retries + 2):
                timeout = self._timeout(deadline)
                if timeout == 0.0:
                    log.warning("%s call not started (attempt %d): too close to the deadline", kind, attempt)
                    break
                attempts += 1
                try:
                    if timeout is None:
                        content = self.client.chat(system, user, images, schema)
                    else:
                        content = self.client.chat(system, user, images, schema, timeout=timeout)
                except VlmCallError as exc:
                    log.warning("%s call failed (attempt %d): %s", kind, attempt, exc)
                    continue
                result = parse(content)
                if result is not None:
                    return result
                log.warning("%s answer not usable (attempt %d): %s", kind, attempt, str(content)[:200])
            self.stats["failed" if attempts else "skipped"] += 1
            return None
        finally:
            elapsed = time.monotonic() - started
            self.stats["seconds"] = round(self.stats["seconds"] + elapsed, 3)
            log.debug("%s call with %d images took %.1f s", kind, len(images), elapsed)

    def facts(self, frames: list[Frame], index: int, start: float, end: float, *, deadline: float | None = None) -> FactWindow | None:
        """One VLM call. Returns None if every attempt fails, the answer cannot be parsed or the deadline is too close.

        `deadline` is a time.monotonic() value; each attempt's timeout is cut so that it ends before it."""
        if not frames:
            log.warning("facts window %d has no frames", index)
            return None
        images = [f.image for f in frames]
        parsed = self._ask("facts", FACTS_SYSTEM, build_facts_prompt(frames), images, facts_schema(len(frames)),
                           lambda content: parse_facts(content, len(frames)), deadline)
        if parsed is None:
            return None
        flags, people, change_frame = parsed
        return FactWindow(
            index=index,
            start=start,
            end=end,
            frame_times=tuple(f.t for f in frames),
            flags=flags,
            people=people,
            change_frame=change_frame,
            motion=round(window_motion(images), 4),
        )

    def verify(
        self, frames: list[Frame], norms: list[Norm], refs: list[CaseRef], type_title: str | None, *, deadline: float | None = None
    ) -> Verification | None:
        """One VLM call. Returns None on failure or when the deadline is too close (see facts)."""
        if not frames or (not type_title and not norms):
            log.warning("verification needs frames and a violation type or norms (frames=%d, norms=%d)", len(frames), len(norms))
            return None
        prompt = build_verify_prompt(frames, norms, refs, type_title)
        images = [r.image_b64 for r in refs] + [f.image for f in frames]
        registry = bool(type_title)
        return self._ask("verify", VERIFY_SYSTEM, prompt, images, verify_schema(len(frames), len(norms)),
                         lambda content: parse_verification(content, len(frames), len(norms), registry), deadline)
