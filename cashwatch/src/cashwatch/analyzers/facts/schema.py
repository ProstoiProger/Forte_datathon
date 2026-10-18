"""Data passed between the stages of the facts analyzer.

Stages: perception (VLM -> FactWindow), case base (FactWindow -> Candidate, CaseRef), regulation search
(text -> Norm), verification (VLM -> Verification). Every stage depends only on these types, so each can be
replaced or scaled out (for example, perception on another GPU host) without touching the others.
"""

from dataclasses import dataclass, field

# The states the VLM reports per window. Same keys and wording as the empirical case memory
# (knowledge/src/forte_kb/cases/annotate.py, CONCEPTS), so runtime facts and stored facts are comparable.
FLAG_DESCRIPTIONS: dict[str, str] = {
    "cashier_at_workplace": "the cashier sits or stands at the cashier desk",
    "cashier_leaving": "the cashier walks away from the desk or is at/through the door",
    "cash_on_desk": "banknotes or cash bundles lie on the desk",
    "cash_in_hands": "the cashier holds banknotes",
    "safe_open": "the door of the metal safe/cabinet is open",
    "door_open": "the room door is open",
    "phone_in_hand": "the cashier holds a mobile phone",
    "drinking": "the cashier holds a cup or drink, or drinks",
    "personal_items_on_desk": "a coat, handbag or other personal bag is on the desk",
    "collection_bag": "a cash-collection bag is visible",
    "visitor_present": "any person other than the cashier is in the room",
}
FLAG_KEYS: tuple[str, ...] = tuple(FLAG_DESCRIPTIONS)
PEOPLE_DESCRIPTION = "integer: the largest number of persons inside the room in any frame"

OTHER_TYPE_ID = "other"  # a confirmed violation whose type is not in the rule registry


@dataclass(frozen=True)
class FactWindow:
    """What the VLM saw in one time window. A flag is True if the state holds in any frame of the window."""

    index: int
    start: float
    end: float
    frame_times: tuple[float, ...]
    flags: dict[str, bool]
    people: int
    change_frame: int = 0  # 1-based frame of the largest visible change, 0 if none
    motion: float = 0.0  # share of pixels that changed between consecutive frames (0..1)


@dataclass(frozen=True)
class Candidate:
    """A window the case base considers suspicious."""

    window: FactWindow
    type_id: str | None  # registry type the rule points to; None = unusual situation without a rule
    rule: str  # name of the rule that fired, or "unusual"
    score: float  # higher = more suspicious; comparable across candidates of one video


@dataclass(frozen=True)
class CaseRef:
    """A reference example from the training videos, shown to the VLM next to the candidate."""

    label: str  # "violation" or "normal"
    type_id: str | None
    image_b64: str  # JPEG, base64 without a data URI prefix
    description: str  # short Russian description of what happens
    source: str  # "X-24#e1" or "X-25@012.00"
    group: str  # camera setup; references from the scored video's setup are excluded during evaluation


@dataclass(frozen=True)
class Norm:
    """A regulation the candidate is checked against."""

    title: str  # violation type title (registry) or document title (search)
    citation: str  # "п. 22, Правила ... (постановление НБРК № 231)"
    quote: str  # verbatim text of the clause
    url: str = ""
    source: str = "registry"  # "registry" (verified) or "rag" (found by search, needs human check)


@dataclass(frozen=True)
class Verification:
    """The VLM's answer for one candidate."""

    confirmed: bool
    frame: int  # 1-based index into the frames shown for the candidate, 0 if none
    description: str  # Russian, what is visible
    confidence: float
    title: str | None = None  # short Russian title when the violation type is not in the registry
    norm_index: int | None = None  # 1-based index of the norm that applies, when several were offered
    extra: dict = field(default_factory=dict)
