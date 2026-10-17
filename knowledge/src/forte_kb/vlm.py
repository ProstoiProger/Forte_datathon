"""Minimal Ollama client for the local VLM."""

import base64
import json
import time
from pathlib import Path

import httpx


class Ollama:
    def __init__(self, url: str, model: str, num_ctx: int = 8192, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.client = httpx.Client(timeout=timeout)

    def loaded(self) -> list[str]:
        r = self.client.get(f"{self.url}/api/ps")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def unload_all(self) -> list[str]:
        """Free the GPU before another model is loaded (keep_alive=0 evicts immediately)."""
        names = self.loaded()
        for name in names:
            self.client.post(f"{self.url}/api/generate", json={"model": name, "keep_alive": 0}).raise_for_status()
        return names

    def chat_json(self, prompt: str, images: list[Path], schema: dict, system: str | None = None,
                  keep_alive: str = "30m") -> tuple[dict | None, str, float]:
        """Returns (parsed JSON or None, raw text, seconds)."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt,
                         "images": [base64.b64encode(Path(p).read_bytes()).decode() for p in images]})
        body = {"model": self.model, "stream": False, "think": False, "format": schema, "keep_alive": keep_alive,
                "options": {"temperature": 0, "num_ctx": self.num_ctx}, "messages": messages}
        t0 = time.time()
        r = self.client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        return parsed, raw, time.time() - t0
