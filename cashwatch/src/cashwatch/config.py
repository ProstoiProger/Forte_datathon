from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CASHWATCH_", env_file=".env", extra="ignore")

    api_keys: str
    data_dir: Path = Path("var")
    analyzer: str = "stub"
    stub_mode: Literal["peak", "none"] = "peak"
    max_upload_mb: float = 200.0
    time_budget_sec: float = 240.0
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
