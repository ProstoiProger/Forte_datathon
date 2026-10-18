from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ProgressFn = Callable[[float], None]


class AnalyzerError(Exception):
    pass


@dataclass(frozen=True)
class EvidenceRef:
    timestamp_sec: float
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Event:
    type_id: str
    timecode_sec: float
    confidence: float
    observation: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    votes: int = 1
    coverage: int = 1
    raw_confidence: float | None = None
    window: int = -1
    # For a type outside the rule registry: its title and the norm found by regulation search
    # (keys: title, citation, quote, url, source). Registry types take title and citations from the registry.
    title: str | None = None
    norm: dict[str, str] | None = None


class Analyzer(Protocol):
    name: str
    default_threshold: float

    def analyze(self, video_path: Path, progress: ProgressFn, deadline: float) -> list[Event]: ...


class NullAnalyzer:
    name = "null"
    default_threshold = 0.0

    def analyze(self, video_path: Path, progress: ProgressFn, deadline: float) -> list[Event]:
        progress(1.0)
        return []

    def check(self) -> tuple[bool, str]:
        return True, "null analyzer"
