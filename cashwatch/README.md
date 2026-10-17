# cashwatch

Local video analysis service that flags visible cash-desk discipline violations in cashier recordings.
It returns a verdict, a timecode, evidence frames and the regulation the event relates to.
All models run inside the service; no cloud AI APIs are called while processing a video.

## Run

```bash
uv sync
cp .env.example .env
uv run cashwatch
```

Set `CASHWATCH_API_KEYS` in `.env` before starting (comma-separated for several keys).
The web UI is served at `http://localhost:8000/`, the API under `/api/v1`.

## API

All `/api/v1` endpoints except `/health` require the `X-API-Key` header.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/analyze` | Upload a `.mov` file in the multipart field `file`. Returns `job_id`. |
| GET | `/api/v1/jobs/{job_id}` | Status `processing`, `done` or `failed`, and the result when done. |
| GET | `/api/v1/health` | Liveness, analyzer name and queue length. |

```bash
curl -X POST http://localhost:8000/api/v1/analyze -H "X-API-Key: <key>" -F "file=@video.mov"
curl http://localhost:8000/api/v1/jobs/<job_id> -H "X-API-Key: <key>"
```

Result when done:

```json
{
  "job_id": "3f2a9c1e0b7d4a55",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Violation type",
      "timecode_sec": 44.2,
      "description": "What is visible and why it may be a violation",
      "evidence_frames": [{"timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..."}]
    }
  ]
}
```

While a job is queued or running the response is `{"job_id", "status": "processing", "progress"}`.
A failed job returns `{"job_id", "status": "failed", "error"}`.
Set `CASHWATCH_API_EXTRA_FIELDS=true` to add `type_id`, `confidence`, `observation` and `rule` to each violation.

Responses: `401` bad or missing key, `404` unknown job, `413` file too large, `415` wrong extension,
`422` missing `file` field, `503` queue full.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CASHWATCH_API_KEYS` | required | Accepted API keys, comma-separated |
| `CASHWATCH_DATA_DIR` | `var` | Uploads and job results |
| `CASHWATCH_ANALYZER` | `stub` | Analyzer backend |
| `CASHWATCH_STUB_MODE` | `peak` | `peak` reports the motion peak, `none` reports nothing |
| `CASHWATCH_MAX_UPLOAD_MB` | `200` | Upload size limit |
| `CASHWATCH_TIME_BUDGET_SEC` | `240` | Processing budget counted from upload |
| `CASHWATCH_QUEUE_MAX` | `64` | Maximum queued jobs |
| `CASHWATCH_API_EXTRA_FIELDS` | `false` | Extra fields in API responses |
| `CASHWATCH_UI_ENABLED` | `true` | Serve the web UI and its keyless endpoints under `/ui/api` |
| `CASHWATCH_KEEP_UPLOADS` | `false` | Keep uploaded videos after processing |

## Design

- One background worker processes jobs in submission order, so a single GPU is never shared between jobs.
- Job state is written to `var/jobs/<job_id>.json` on every transition; finished results survive restarts and
  unfinished uploads are re-queued on startup.
- Uploaded videos are deleted after processing unless `CASHWATCH_KEEP_UPLOADS=true`.
- Analyzers return events (type, time, confidence, observation, evidence times). Evidence frames are decoded,
  encoded as JPEG and attached by the service, so every analyzer produces the same output format.
- Violation types and their legal basis live in `src/cashwatch/rules/registry.json`. A citation is shown only
  when it is marked `verified` and carries a verbatim quote from the official text.

## Analyzer

`CASHWATCH_ANALYZER=vlm` runs Qwen3-VL through a local Ollama server.

- Frames are sampled at `CASHWATCH_VLM_FPS` (default 2) and grouped into overlapping windows
  (`CASHWATCH_VLM_WINDOW_SEC=6`, `CASHWATCH_VLM_STRIDE_SEC=3`, at most `CASHWATCH_VLM_MAX_FRAMES=12`).
- The prompt lists every violation type from `registry.json`; the answer is constrained by a JSON schema whose
  `type_id` enum is the same list, so the model can only return known types.
- Long videos are capped at `CASHWATCH_VLM_MAX_WINDOWS` model calls (default 20): stride and, if needed, window length
  grow so the whole video is still covered, which bounds processing time.
- Detections of the same type closer than `CASHWATCH_VLM_MERGE_GAP_SEC` are merged. With `CASHWATCH_VLM_SCORE=agreement`
  (default) the event score is the model confidence scaled by how many of the windows that saw the moment reported it
  (`score = confidence * (0.5 + 0.5 * votes / coverage)`); `max` keeps the raw confidence.
- `CASHWATCH_VLM_REFERENCE_DIR` adds up to `CASHWATCH_VLM_MAX_REFERENCES` example frames (one per type plus normal work)
  before the frames to analyze; build them with `scripts/build_references.py`.
- A failed call is retried once; failed windows are skipped. If every call fails, the fallback analyzer
  (`CASHWATCH_FALLBACK=none|stub`) produces the result instead of a failed job.
- Windows stop `CASHWATCH_VLM_SAFETY_SEC` before the job deadline; the result then covers the processed part.
- Detections below `CASHWATCH_MIN_CONFIDENCE` (default 0.5 for `vlm`) are dropped before the verdict.
- `/api/v1/health` reports whether Ollama is reachable and the model is pulled.

```bash
ollama pull qwen3-vl:8b
CASHWATCH_ANALYZER=vlm uv run cashwatch
```

## Evaluation

```bash
uv run python scripts/build_labels.py
uv run python scripts/evaluate.py run --name vlm8b --analyzer vlm --holdout s11,s12
uv run python scripts/evaluate.py score --run runs/vlm8b --holdout s11,s12
```

`build_labels.py` converts the organizers' spreadsheet into `data/labels.json`: duplicate files are merged into one
video with several events, every video gets a camera setup group, every event a hypothesis of its type
(`type_status` is `likely` or `uncertain` until confirmed by watching).

`evaluate.py run` caches predictions per video in `runs/<name>/predictions.jsonl` (rerunning skips finished videos,
`--restart` clears them) and then scores. `score` reports:

- detection: accuracy, recall and false-positive rate of the per-video verdict;
- localization: share of labeled events matched within `--tol` seconds (default 2), within 1 s, and the mean error;
- classification: share of labeled events whose matched detection has the right type;
- a proxy of the 55 points for these three criteria (25 / 15 / 15);
- three views of the threshold: tuned on all videos (optimistic), leave-one-setup-out (honest) and tuned on all
  setups except `--holdout`, scored on the holdout;
- a threshold sweep and `runs/<name>/videos.csv` with per-video issues (missed, false positive, offset, wrong type).

Settings can be overridden per run, for example `--vlm-model qwen3-vl:30b --vlm-stride-sec 4`.

`--transform flip|crop|warp|rotate|dark|lowres` evaluates on transformed copies of the videos (cached in
`data/transformed`) to estimate robustness to another camera angle. When reference frames are used, references from
the same camera setup as the scored video are left out.

```bash
uv run python scripts/build_references.py
uv run python scripts/run_queue.py experiments.json
uv run python scripts/run_queue.py --leaderboard-only
```

`run_queue.py` runs the experiments listed in a JSON file (see `experiments.example.json`) one after another, skips
runs that already have `metrics.json`, logs to `runs/queue.log` and writes `runs/leaderboard.md` sorted by the
leave-one-setup-out score.

## Legal basis

Five acts are covered clause by clause (388 clauses): NBRK Rules No. 231 (cash operations and collection),
No. 14 (premises), No. 49 (currency exchange), No. 230 (banknote payability) and Government Resolution No. 305
(anti-terrorism protection). Every clause is assigned a visibility tier: A (directly visible), B (partly visible),
DEF (definition) or C (not verifiable from video). Violation types cite only tier A and B clauses.

```bash
uv run python scripts/legal_fetch.py
uv run python scripts/build_rules.py
```

`legal_fetch.py` downloads the texts into `data/legal/` and splits them into clauses. `build_rules.py` checks
that every quote in `legal/annotations.json` appears verbatim in its clause, then writes
`src/cashwatch/rules/catalog.json`, `src/cashwatch/rules/registry.json` and `docs/RULES.md`.
Quotes are additionally checked against the rendered text of adilet.zan.kz; the result is recorded in
`legal/annotations.json` under `adilet_verified`.

## Tests

```bash
uv run pytest
```

`tests/test_contract.py` checks authentication, upload validation, the job lifecycle and the exact response
schema. Set `CASHWATCH_SAMPLE_VIDEO=<path to .mov>` to also run a real video through the service.

`scripts/smoke.py` runs the same schema check against a deployed instance:

```bash
uv run python scripts/smoke.py --url https://<host> --key <key> --file video.mov
```
