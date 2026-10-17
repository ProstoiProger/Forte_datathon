from pathlib import Path

from cashwatch.analyzers.base import Analyzer, AnalyzerError, Event, EvidenceRef, NullAnalyzer
from cashwatch.config import Settings


def build_analyzer(settings: Settings) -> Analyzer:
    if settings.analyzer == "stub":
        from cashwatch.analyzers.stub import MotionStubAnalyzer

        return MotionStubAnalyzer(mode=settings.stub_mode)
    if settings.analyzer == "null":
        return NullAnalyzer()
    if settings.analyzer == "vlm":
        from cashwatch.analyzers.fallback import FallbackAnalyzer
        from cashwatch.analyzers.stub import MotionStubAnalyzer
        from cashwatch.analyzers.vlm import OllamaClient, VlmAnalyzer, load_references

        client = OllamaClient(
            settings.vlm_url,
            settings.vlm_model,
            timeout=settings.vlm_timeout_sec,
            num_ctx=settings.vlm_num_ctx,
            keep_alive=settings.vlm_keep_alive,
        )
        references = []
        if settings.vlm_reference_dir:
            references = load_references(Path(settings.vlm_reference_dir), settings.vlm_reference_max_side)
        primary = VlmAnalyzer(
            client,
            fps=settings.vlm_fps,
            window_sec=settings.vlm_window_sec,
            stride_sec=settings.vlm_stride_sec,
            max_frames=settings.vlm_max_frames,
            max_side=settings.vlm_max_side,
            jpeg_quality=settings.vlm_jpeg_quality,
            retries=settings.vlm_retries,
            merge_gap_sec=settings.vlm_merge_gap_sec,
            safety_sec=settings.vlm_safety_sec,
            max_windows=settings.vlm_max_windows,
            score_mode=settings.vlm_score,
            references=references,
            max_references=settings.vlm_max_references,
        )
        secondary = MotionStubAnalyzer(mode="peak") if settings.fallback == "stub" else NullAnalyzer()
        return FallbackAnalyzer(primary, secondary)
    raise ValueError(f"unknown analyzer {settings.analyzer!r}")


def threshold_for(settings: Settings, analyzer: Analyzer) -> float:
    if settings.min_confidence is not None:
        return settings.min_confidence
    return float(getattr(analyzer, "default_threshold", 0.0))


__all__ = ["Analyzer", "AnalyzerError", "Event", "EvidenceRef", "NullAnalyzer", "build_analyzer", "threshold_for"]
