"""Qwen3-VL-Embedding wrapper.

Adapted from scripts/qwen3_vl_embedding.py of Qwen/Qwen3-VL-Embedding-2B (Apache-2.0): same chat-template
input format, last-token pooling and L2 normalization. Changes: batching, MRL truncation, a frame list is
used as given (the reference resamples every list to 64 frames by duplication), and explicit unload so the
embedder and the large VLM never share the GPU.
"""

import gc
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
MAX_LENGTH = 8192

# Instructions are in English as the model card recommends. Clip-to-clip search uses CLIP on both sides.
CLIP = "Represent this top-down CCTV footage of a bank cash room for finding visually similar cashier situations."
TEXT_TO_CLIP = "Find top-down CCTV footage of a bank cash room that shows the described situation."
CARD = "Represent this description of a cashier situation in a bank cash room."
TEXT_TO_CARD = "Find descriptions of bank cash-room cases that match this observation."


@dataclass
class Item:
    """One input: text, an image, a list of frames (treated as a video), or a combination."""
    text: str | None = None
    image: Image.Image | Path | None = None
    frames: list[Image.Image | Path] | None = None
    instruction: str | None = None


def _image(x: Image.Image | Path) -> Image.Image:
    return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


class Embedder:
    def __init__(self, model_path: str | Path, dim: int = 2048, dtype: str = "bfloat16", device: str | None = None):
        self.model_path = str(model_path)
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> "Embedder":
        if self.model is not None:
            return self
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            _checkpoint_conversion_mapping = {}
            accepts_loss_kwargs = False

            def __init__(self, config):
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()

            def forward(self, **kwargs):
                return self.model(**kwargs).last_hidden_state

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3VLForEmbedding.from_pretrained(
            self.model_path, dtype=getattr(torch, self.dtype), attn_implementation="sdpa", local_files_only=True
        ).to(device).eval()
        self.processor = Qwen3VLProcessor.from_pretrained(self.model_path, padding_side="right", local_files_only=True)
        log.info("embedder loaded on %s", device)
        return self

    def unload(self) -> None:
        if self.model is None:
            return
        import torch
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _conversation(item: Item) -> list[dict]:
        instruction = (item.instruction or "Represent the user's input.").strip()
        if not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        content: list[dict[str, Any]] = []
        if item.frames:
            frames = [_image(f) for f in item.frames]
            if len(frames) % 2:  # the vision tower packs frames in temporal pairs
                frames.append(frames[-1])
            content.append({"type": "video", "video": frames, "total_pixels": MAX_TOTAL_PIXELS})
        if item.image is not None:
            content.append({"type": "image", "image": _image(item.image),
                            "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS})
        if item.text:
            content.append({"type": "text", "text": item.text})
        if not content:
            content.append({"type": "text", "text": "NULL"})
        return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content}]

    def _encode_batch(self, items: list[Item]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils.vision_process import process_vision_info

        conversations = [self._conversation(it) for it in items]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
        videos = metadata = None
        if video_inputs is not None:
            videos, metadata = (list(x) for x in zip(*video_inputs))
        inputs = self.processor(text=text, images=images, videos=videos, video_metadata=metadata, truncation=True,
                                max_length=MAX_LENGTH, padding=True, do_resize=False, return_tensors="pt",
                                **video_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            hidden = self.model(**inputs)
        mask = inputs["attention_mask"]
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        pooled = pooled[:, : self.dim].float()
        return F.normalize(pooled, p=2, dim=-1).cpu().numpy()

    def encode(self, items: list[Item], batch_size: int = 8, progress: str | None = None) -> np.ndarray:
        self.load()
        out = []
        for i in range(0, len(items), batch_size):
            out.append(self._encode_batch(items[i:i + batch_size]))
            if progress:
                log.info("%s %d/%d", progress, min(i + batch_size, len(items)), len(items))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)
