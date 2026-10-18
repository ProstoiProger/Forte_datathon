"""Declarative rules over the facts of one video.

A rule names a registry violation type (or None) and a predicate over one window in the context of its whole video
(previous/next windows, how long a state lasts, what is usual for this video). Rules only say when they fire; how
much a firing counts is learned from the case base (see store.py): a rule that fires as often in normal work as in
labeled violations gets no weight. Facts are window-level (a flag is on if it holds in any frame of the window), so
rules combine flags of the same window, not of the same frame.

Runtime windows may have gaps (static windows skipped under a time budget, failed perception calls) and may be wider
than the 4 s the weights were learned on (4 frames per window either way). The context therefore works in seconds:
- windows are consecutive only if the next one starts within GAP_TOLERANCE of the previous end; runs break at gaps;
- a duration rule needs at least two consecutive perceived windows and LONG_SECONDS of covered time, so one wide
  window with one brief flag is not "long";
- claims about the end of the clip (not back before the end, still open at the end) need the window covering the
  true end (pass the clip duration); "comes in" needs the window covering the start; state changes (a visitor
  appears or leaves) compare with the nearest perceived neighbour, since skipped static windows carry the state over;
- "usual for this video" is the share of perceived windows (one video has one window length, widened or not).
Known limits: wider windows make any-frame flags and the maximum of `people` fire more often than in the 4 s training
windows, so learned weights overstate brief flags there; under motion-ordered perception the perceived windows are
the busy ones, which raises the usual share of transient states.

Design notes (training data, reviewer notes in knowledge/cases_review.json):
- single facts are normal work: phone in hand (12/19 clean videos), door open (13/19), cashier leaving (10/19),
  a safe opened for a few seconds (10/19);
- combinations and durations are not: cash on the desk while the cashier walks away, three or more people, a
  visitor appearing in a room where visits are rare, personal items or a drink kept on the desk for a long time;
- brief "drinking" in clean videos is a perception error (a card or cash containers held near the face, X-16, X-25),
  so only sustained drinking is a rule; a brief one only counts as an unusual fact;
- with a collection bag in view the VLM reports the bag or the collector's things as personal items (X-1, X-3, X-16);
- "left open" types are also checked at the end of the clip: the VLM misses a glass door or the safe door left open
  while the cashier sits at the desk (X-20, X-32), so these rules matter with better perception.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

from cashwatch.analyzers.facts.schema import FLAG_KEYS, FactWindow

UNUSUAL_SHARE = 1 / 3  # a flag on in at most this share of a video's perceived windows is unusual for that video
UNUSUAL_MIN_WINDOWS = 3  # shorter videos have no notion of "usual"
LONG_SECONDS = 6.0  # a state "lasts" when it holds longer than one 4 s window ...
LONG_MIN_WINDOWS = 2  # ... over at least this many consecutive perceived windows
GAP_TOLERANCE = 0.5  # s: closer windows are consecutive; a window ending this close to the end covers it
RARE_VISITS = 0.5  # visitors are rare in a video when present in at most this share of its windows
CROWD = 3  # persons in the room

UNUSUAL_PREFIX = "unusual:"
UNUSUAL_PEOPLE = UNUSUAL_PREFIX + "people"
UNUSUAL_FLAGS: tuple[str, ...] = tuple(k for k in FLAG_KEYS if k != "cashier_at_workplace")  # absence = leaving rules


class VideoContext:
    """The perceived windows of one video in time order, with per-video statistics used by the predicates.

    `duration` is the true clip length; without it the last perceived window is taken as the end of the clip.
    """

    def __init__(self, windows: Sequence[FactWindow], duration: float | None = None) -> None:
        self.windows = tuple(sorted(windows, key=lambda w: w.start))
        self.duration = duration
        n = len(self.windows)
        self.share = {k: (sum(bool(w.flags.get(k)) for w in self.windows) / n if n else 0.0) for k in FLAG_KEYS}
        self.usual_people = float(median(w.people for w in self.windows)) if n else 0.0
        self.covers_start = n > 0 and self.windows[0].start <= GAP_TOLERANCE
        if duration is None:
            self.covers_end = n > 0
        else:
            self.covers_end = n > 0 and self.windows[-1].end >= duration - GAP_TOLERANCE

    def __len__(self) -> int:
        return len(self.windows)

    def on(self, i: int, key: str) -> bool:
        return 0 <= i < len(self.windows) and bool(self.windows[i].flags.get(key))

    def people(self, i: int) -> int:
        return self.windows[i].people if 0 <= i < len(self.windows) else 0

    def consecutive(self, i: int) -> bool:
        """Window i+1 follows window i without a gap."""
        if not 0 <= i < len(self.windows) - 1:
            return False
        return self.windows[i + 1].start - self.windows[i].end <= GAP_TOLERANCE

    def run(self, i: int, key: str) -> tuple[int, float]:
        """(windows, seconds) of the gap-free stretch of consecutive windows with `key` on that contains window i."""
        if not self.on(i, key):
            return 0, 0.0
        first, last = i, i
        while self.consecutive(first - 1) and self.on(first - 1, key):
            first -= 1
        while self.consecutive(last) and self.on(last + 1, key):
            last += 1
        return last - first + 1, self.windows[last].end - self.windows[first].start

    def lasts(self, i: int, key: str) -> bool:
        windows, seconds = self.run(i, key)
        return windows >= LONG_MIN_WINDOWS and seconds >= LONG_SECONDS

    def later(self, i: int, key: str) -> bool:
        return any(self.on(j, key) for j in range(i + 1, len(self.windows)))

    def earlier(self, i: int, key: str) -> bool:
        return any(self.on(j, key) for j in range(i))

    def is_end(self, i: int) -> bool:
        """Window i covers the true end of the clip and is not also its start."""
        return i == len(self.windows) - 1 and self.covers_end and self.windows[i].start > GAP_TOLERANCE

    def unusual(self, i: int, key: str) -> bool:
        return len(self.windows) >= UNUSUAL_MIN_WINDOWS and self.on(i, key) and self.share[key] <= UNUSUAL_SHARE


Predicate = Callable[[VideoContext, int], bool]


@dataclass(frozen=True)
class Rule:
    name: str
    type_id: str | None
    description: str
    when: Predicate


def _away(c: VideoContext, i: int) -> bool:
    return c.on(i, "cashier_leaving") or not c.on(i, "cashier_at_workplace")


def left_workplace(c: VideoContext, i: int) -> bool:
    return c.covers_end and c.on(i, "cashier_leaving") and not c.later(i, "cashier_at_workplace")


def cash_unattended(c: VideoContext, i: int) -> bool:
    return c.on(i, "cash_on_desk") and _away(c, i)


def safe_open_unattended(c: VideoContext, i: int) -> bool:
    return c.on(i, "safe_open") and _away(c, i)


def safe_open_at_end(c: VideoContext, i: int) -> bool:
    return c.is_end(i) and c.on(i, "safe_open")


def crowd(c: VideoContext, i: int) -> bool:
    return c.people(i) >= CROWD


def visitor_appears(c: VideoContext, i: int) -> bool:
    rare = c.share["visitor_present"] <= RARE_VISITS
    return rare and c.on(i, "visitor_present") and not c.on(i - 1, "visitor_present")


def entered_door_open(c: VideoContext, i: int) -> bool:
    arrives = c.covers_start and not c.earlier(i, "cashier_at_workplace") and c.later(i, "cashier_at_workplace")
    return arrives and c.on(i, "door_open") and c.on(i, "cashier_leaving")


def door_open_long(c: VideoContext, i: int) -> bool:
    return c.on(i, "cashier_at_workplace") and c.lasts(i, "door_open")


def door_open_at_end(c: VideoContext, i: int) -> bool:
    return c.is_end(i) and c.on(i, "door_open") and c.on(i, "cashier_at_workplace")


def personal_items(c: VideoContext, i: int) -> bool:
    return c.on(i, "personal_items_on_desk") and not c.on(i, "collection_bag")


def personal_items_long(c: VideoContext, i: int) -> bool:
    return personal_items(c, i) and c.lasts(i, "personal_items_on_desk")


def lasts(key: str) -> Predicate:
    return lambda c, i: c.lasts(i, key)


def present(key: str) -> Predicate:
    return lambda c, i: c.on(i, key)


def _visitor_gone_next(c: VideoContext, i: int) -> bool:
    return c.on(i, "visitor_present") and i + 1 < len(c) and not c.on(i + 1, "visitor_present")


def visitor_leaves(c: VideoContext, i: int) -> bool:
    return _visitor_gone_next(c, i) and c.on(i + 1, "cashier_at_workplace")


def collector_leaves(c: VideoContext, i: int) -> bool:
    return _visitor_gone_next(c, i) and (c.on(i, "collection_bag") or c.on(i - 1, "collection_bag"))


RULES: tuple[Rule, ...] = (
    Rule("left_workplace", "workplace_abandoned",
         "the cashier walks away and is not back at the desk before the end of the clip", left_workplace),
    Rule("cash_unattended", "cash_unsecured",
         "cash lies on the desk while the cashier walks away or is not at the desk", cash_unattended),
    Rule("safe_open_unattended", "safe_left_open",
         "the safe is open while the cashier walks away or is not at the desk", safe_open_unattended),
    Rule("safe_open_long", "safe_left_open", f"the safe stays open for {LONG_SECONDS:.0f}+ s", lasts("safe_open")),
    Rule("safe_open_at_end", "safe_left_open", "the safe is still open when the clip ends", safe_open_at_end),
    Rule("crowd", "unauthorized_entry", f"{CROWD} or more persons in the room", crowd),
    Rule("visitor_appears", "unauthorized_entry",
         "a person enters a room where visitors are rare in this video", visitor_appears),
    Rule("entered_door_open", "door_left_open",
         "the cashier comes in through the open door and then works at the desk (the door may stay open)",
         entered_door_open),
    Rule("door_open_long", "door_left_open",
         f"the door stays open for {LONG_SECONDS:.0f}+ s while the cashier is at the desk", door_open_long),
    Rule("door_open_at_end", "door_left_open", "the door is still open when the clip ends, the cashier at the desk",
         door_open_at_end),
    Rule("personal_items", "personal_items", "a coat or bag lies on the desk (no collection bag in view)",
         personal_items),
    Rule("personal_items_long", "personal_items",
         f"a coat or bag stays on the desk for {LONG_SECONDS:.0f}+ s (no collection bag in view)", personal_items_long),
    Rule("drinking_long", "food_drink", f"the cashier drinks or holds a drink for {LONG_SECONDS:.0f}+ s",
         lasts("drinking")),
    Rule("phone", "phone_use", "the cashier holds a mobile phone", present("phone_in_hand")),
    Rule("collector_leaves", "collector_handover",
         "a visitor leaves right after a collection bag was visible", collector_leaves),
    Rule("visitor_leaves", "cash_handling",
         "the client leaves while the cashier stays at the desk (cash may be handled without the client)",
         visitor_leaves),
)


def unusual_features(c: VideoContext, i: int) -> list[str]:
    """Facts on in window i that are rare for this video (a state that appears briefly), plus more people than usual."""
    out = [UNUSUAL_PREFIX + k for k in UNUSUAL_FLAGS if c.unusual(i, k)]
    if len(c) >= UNUSUAL_MIN_WINDOWS and c.people(i) > c.usual_people:
        out.append(UNUSUAL_PEOPLE)
    return out


UNUSUAL_FEATURES: tuple[str, ...] = tuple(UNUSUAL_PREFIX + k for k in UNUSUAL_FLAGS) + (UNUSUAL_PEOPLE,)


def fire(windows: Sequence[FactWindow], rules: Sequence[Rule] = RULES,
         duration: float | None = None) -> list[frozenset[str]]:
    """Names of the rules and unusual features that fire in each window of one video (sorted by start time)."""
    ctx = VideoContext(windows, duration)
    return [frozenset([r.name for r in rules if r.when(ctx, i)] + unusual_features(ctx, i)) for i in range(len(ctx))]
