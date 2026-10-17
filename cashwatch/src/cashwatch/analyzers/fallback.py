import logging
from pathlib import Path

from cashwatch.analyzers.base import Analyzer, AnalyzerError, Event, ProgressFn

log = logging.getLogger("cashwatch.fallback")


class FallbackAnalyzer:
    def __init__(self, primary: Analyzer, secondary: Analyzer) -> None:
        self.primary = primary
        self.secondary = secondary
        self.name = primary.name
        self.default_threshold = primary.default_threshold
        self.fallbacks = 0

    def analyze(self, video_path: Path, progress: ProgressFn, deadline: float) -> list[Event]:
        try:
            return self.primary.analyze(video_path, progress, deadline)
        except AnalyzerError as exc:
            self.fallbacks += 1
            log.warning("primary analyzer %s failed (%s), using %s", self.primary.name, exc, self.secondary.name)
            return self.secondary.analyze(video_path, progress, deadline)

    def check(self) -> tuple[bool, str]:
        check = getattr(self.primary, "check", None)
        return check() if check else (True, self.primary.name)
