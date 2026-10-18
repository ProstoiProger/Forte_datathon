from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CASHWATCH_", env_file=".env", extra="ignore")

    api_keys: str
    data_dir: Path = Path("var")
    analyzer: str = "stub"
    stub_mode: Literal["peak", "none"] = "peak"
    max_upload_mb: float = 200.0
    time_budget_sec: float = 240.0
    fair_share_budget: bool = True  # split the time left with the jobs waiting in the queue
    min_job_sec: float = 25.0  # enough to decode and perceive a few windows
    max_job_sec: float | None = 100.0  # a job that starts alone leaves time for uploads arriving right after it
    queue_max: int = 64
    api_extra_fields: bool = False
    ui_enabled: bool = True
    keep_uploads: bool = False
    allowed_extensions: tuple[str, ...] = (".mov",)
    ui_allowed_extensions: tuple[str, ...] = (".mov", ".mp4", ".m4v")
    host: str = "0.0.0.0"
    port: int = 8000
    min_confidence: float | None = None
    fallback: Literal["none", "stub"] = "none"
    vlm_url: str = "http://127.0.0.1:11434"
    vlm_model: str = "qwen3-vl:8b"
    vlm_fps: float = 2.0
    vlm_window_sec: float = 6.0
    vlm_stride_sec: float = 3.0
    vlm_max_frames: int = 12
    vlm_max_side: int = 848
    vlm_jpeg_quality: int = 85
    vlm_timeout_sec: float = 90.0
    vlm_retries: int = 1
    vlm_num_ctx: int = 16384
    vlm_merge_gap_sec: float = 4.0
    vlm_safety_sec: float = 10.0
    vlm_keep_alive: str = "60m"
    vlm_max_windows: int = 20
    vlm_score: Literal["agreement", "max"] = "agreement"
    vlm_reference_dir: str | None = None
    vlm_max_references: int = 6
    vlm_reference_max_side: int = 512
    # facts analyzer: VLM facts per window -> case-base candidates -> VLM verification against one norm
    facts_window_sec: float = Field(4.0, gt=0)
    facts_frames: int = Field(4, ge=1)
    facts_verify_frames: int = Field(8, ge=1)
    facts_verify_context_sec: float = Field(3.0, ge=0)
    facts_max_windows: int = Field(24, ge=1)
    facts_min_windows: int = Field(3, ge=1)  # perceived even when the verification reserve has to be spent
    facts_max_candidates: int = Field(2, ge=1)
    facts_max_verify: int = Field(2, ge=0)
    facts_est_call_sec: float = Field(6.0, gt=0)  # measured on the RTX 5090 Laptop with qwen3.8:27b, 4 frames
    facts_est_verify_sec: float = Field(11.0, gt=0)  # one verification call (8 frames + 2 references)
    facts_verify_reserve_sec: float = Field(25.0, ge=0)
    facts_unverified_confidence: float = Field(0.6, ge=0, le=1)
    facts_keep_rejected_types: str = ""  # comma-separated type ids kept when the verifier rejects them (recall first)
    casebase_dir: str = "data/casebase"
    regsearch_url: str | None = "http://127.0.0.1:8765"
    regsearch_timeout_sec: float = Field(5.0, gt=0)

    @field_validator("api_keys")
    @classmethod
    def _keys_present(cls, value: str) -> str:
        if not [k for k in value.split(",") if k.strip()]:
            raise ValueError("CASHWATCH_API_KEYS must contain at least one key")
        return value

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
