"""Dense vectors of the chunks (Qwen3-VL-Embedding, text mode) in the on-disk store of forte_kb."""

import logging
import time
from pathlib import Path

import numpy as np
from forte_kb.embedder import Embedder, Item
from forte_kb.store import Space

from . import config
from .corpus import Chunk

log = logging.getLogger(__name__)


def embed_chunks(chunks: list[Chunk], embedder: Embedder, batch_size: int = 32) -> np.ndarray:
    """Embeds in order of length (less padding per batch) and returns rows in the original order."""
    texts = [c.index_text() for c in chunks]
    order = np.argsort([len(t) for t in texts])[::-1]  # longest first: an out-of-memory error shows up at once
    vecs = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        vecs[idx] = embedder.encode([Item(text=texts[j], instruction=config.DOC_INSTRUCTION) for j in idx],
                                    batch_size=len(idx))
        done = min(i + batch_size, len(order))
        if (i // batch_size) % 50 == 0 or done == len(order):
            rate = done / max(time.time() - t0, 1e-9)
            log.info("dense %d/%d  %.0f chunks/s  eta %.0f s", done, len(order), rate, (len(order) - done) / rate)
    return vecs


def build(chunks: list[Chunk], embedder: Embedder, store: Path, batch_size: int = 32) -> Space:
    vecs = embed_chunks(chunks, embedder, batch_size)
    meta = [{"id": c.id, "act_id": c.act_id, "kind": c.kind} for c in chunks]
    space = Space("dense", vecs, meta)
    space.save(store)
    return space


def embed_query(embedder: Embedder, query: str) -> np.ndarray:
    return embedder.encode([Item(text=query, instruction=config.QUERY_INSTRUCTION)], batch_size=1)[0]
