"""Paths and model settings. Every value can be overridden with an environment variable of the same name."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FORTE = PROJECT.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


VIDEOS_DIR = _path("KB_VIDEOS_DIR", FORTE / "open" / "videos")
LABELS_XLSX = _path("KB_LABELS_XLSX", FORTE / "open" / "open_labels.xlsx")
CASES_DIR = _path("KB_CASES_DIR", PROJECT / "store" / "cases")
CASHWATCH_REGISTRY = _path("KB_CASHWATCH_REGISTRY", FORTE / "cashwatch" / "src" / "cashwatch" / "rules" / "registry.json")
FRAME_REVIEW_JSON = _path("KB_FRAME_REVIEW_JSON", FORTE.parent / "Datathon" / "runs" / "vlm_bakeoff" / "gt.json")

# A local directory or a Hugging Face repo id already in the local cache; loading never goes online.
EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
EMBED_DIM = int(os.environ.get("KB_EMBED_DIM", "2048"))

OLLAMA_URL = os.environ.get("KB_OLLAMA_URL", "http://127.0.0.1:11434")
VLM_MODEL = os.environ.get("KB_VLM_MODEL", "qwen3.8:27b")
VLM_NUM_CTX = int(os.environ.get("KB_VLM_NUM_CTX", "8192"))
VLM_TIMEOUT_SEC = float(os.environ.get("KB_VLM_TIMEOUT_SEC", "600"))

# Frame store and clip windows. Frames are decoded once at FRAME_FPS; windows are cut from them.
FRAME_FPS = 2.0
WINDOW_SEC = 4.0
WINDOW_STRIDE_SEC = 2.0
CLIP_HALF_SEC = 5.0
