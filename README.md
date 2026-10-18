# ForteBank Datathon 26 — cash-discipline violations from video

A fully local AI service that analyzes a cashier's CCTV recording and returns a verdict, the violation type, a
timecode, evidence frames, a description of what is visible and the Kazakhstan regulation it relates to.
No cloud AI services are called while processing a video.

## Layout

| Folder | What it is | Status |
|---|---|---|
| [`cashwatch/`](cashwatch/README.md) | the service: API with key auth, job queue, web UI, rule registry (15 violation types with verbatim citations checked against adilet.zan.kz), analyzers `facts` (default: VLM facts -> case base -> VLM verification against a norm) and `vlm` (zero-shot) | working |
| [`knowledge/`](knowledge/README.md) | empirical case memory built from the 40 labeled training videos: per-window VLM facts, reviewed events with rules, what normal work looks like, embeddings, leave-one-setup-out evaluation | built offline |
| [`rag/`](rag/README.md) | hybrid retrieval over 313 Kazakhstan banking cash-desk acts: BM25 + dense vectors + reciprocal rank fusion, optional cross-encoder | built offline |
| `npa_bank_kz/` | regulatory corpus (adilet.zan.kz, state of the database 14.10.2026): only `registry.json` and `texts/` are tracked; provenance in [`rag/SOURCES.md`](rag/SOURCES.md) | input of `rag/` |
| `datathon26_vlm_architecture_context_for_claude.md` | original architecture brief (three knowledge layers, open-world design) | reference |
| `docker-compose.yml`, `.env.example` | the whole service in Docker: `rag`, `cashwatch`, public tunnel, optional `ollama`; one project-wide `.env` | deployment |

Presentation notes with experiments and design decisions: [`knowledge/PRESENTATION.md`](knowledge/PRESENTATION.md),
[`rag/PRESENTATION.md`](rag/PRESENTATION.md).

## Not in the repository

- **Organizers' dataset** (`open/`: training videos and `open_labels.xlsx`). The data may be used only within the
  hackathon (FAQ 6) and is never committed. Put it at `open/videos/*.mov` and `open/open_labels.xlsx`.
- **Generated stores** (`knowledge/store/`, `rag/store/`): rebuilt by the commands below; `knowledge/store`
  contains frames of the dataset videos.
- **Model weights**: pulled once, then everything runs offline.

## Run with Docker

```
internet ─► https://<random>.trycloudflare.com ─► tunnel ─► cashwatch:8000 ─► Ollama on the host (GPU, qwen3.8:27b)
                                                                         └─► rag:8765 (CPU, regulation search)
```

```powershell
copy .env.example .env      # then set CASHWATCH_API_KEYS (first key = jury, second = team) and HF_CACHE
docker compose up -d --build
docker compose ps           # cashwatch and rag must be "healthy"
docker compose logs tunnel | findstr trycloudflare    # public URL: UI at /, API at /api/v1
docker compose down
```

- Prerequisites on the host: Docker Desktop, Ollama with `qwen3.8:27b`, the Hugging Face cache with
  `Qwen/Qwen3-VL-Embedding-2B`, the case base bundle `cashwatch/data/casebase/` and the indexes `rag/store/` (see
  Setup). These are mounted read-only; images contain code only and nothing is downloaded at runtime.
- Clean machine without Ollama on the host: `docker compose --profile ollama up -d`, set
  `OLLAMA_URL=http://ollama:11434` in `.env`, then `docker compose exec ollama ollama pull qwen3.8:27b` once.
- The quick-tunnel URL changes whenever the `tunnel` container restarts; do not restart it after handing the URL in.
  A network drop on the host can make Cloudflare delete the quick tunnel (the tunnel logs repeat
  `Register tunnel error ... Tunnel not found` and the URL no longer resolves); it does not recover by itself:
  `docker compose restart tunnel` and hand in the new URL. A named tunnel (Cloudflare account and a domain) keeps a
  fixed URL.
  cashwatch is published on 127.0.0.1:47800 only (an uncommon port, to avoid clashes with other local services), so from outside it is reachable through the tunnel alone.
- Check from outside: `cd cashwatch; uv run python scripts/smoke.py --url <public URL> --key <jury key> --file ..\open\videos\X-1.mov`
  must print `no key -> 401` and `done in ...s` without `FAIL`. Five uploads at once were answered within 244 s
  (limit 300 s): jobs share the time left fairly (`CASHWATCH_FAIR_SHARE_BUDGET`, `CASHWATCH_MIN_JOB_SEC`,
  `CASHWATCH_MAX_JOB_SEC`).
- The web UI (`/`) needs no key, the jury API (`/api/v1`) does; set `CASHWATCH_UI_ENABLED=false` to serve the API only.
- Keep the host awake during the jury run: charger in, sleep "never", lid close does nothing, Windows Update paused,
  Docker Desktop and Ollama started at login.

## Setup (development, builds the data the containers mount)

Requirements: Windows or Linux, an NVIDIA GPU with 24 GB VRAM (developed on an RTX 5090 Laptop), `uv`,
`ffmpeg`/`ffprobe` on PATH, Ollama.

```powershell
ollama pull qwen3.8:27b                                   # primary VLM
# Hugging Face cache (downloaded once): Qwen/Qwen3-VL-Embedding-2B, BAAI/bge-reranker-v2-m3 (optional)

cd knowledge; uv sync; uv run kb cases labels; uv run kb cases frames; uv run kb cases annotate
uv run kb cases cases; uv run kb cases embed; uv run kb cases eval; cd ..        # empirical case memory
cd rag; uv sync; uv run rag build; uv run rag eval; cd ..                       # regulation indexes
cd cashwatch; uv sync; uv run pytest; uv run python scripts/build_casebase.py; cd ..   # service + case base bundle
```

`knowledge` and `rag` pin torch 2.8.0 with CUDA 12.8 wheels (needed for RTX 50xx). `rag` uses the embedder
wrapper and the vector store of `knowledge` as a read-only path dependency.

## Models and licenses

| Model / library | Use | License |
|---|---|---|
| Qwen3.8-27B (Ollama, Q4_K_M) | video facts, event descriptions, analyzer | check the model card of the exact build |
| Qwen3-VL-Embedding-2B | embeddings of clips, frames, texts and regulations | Apache-2.0 |
| BAAI/bge-reranker-v2-m3 | optional reranking | Apache-2.0 |
| bm25s, PyStemmer (Snowball) | BM25 with Russian stemming | MIT; Snowball BSD-3 |
| FastAPI, OpenCV, PyTorch, transformers | service and inference | MIT / Apache-2.0 / BSD |
