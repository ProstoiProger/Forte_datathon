# knowledge

Local knowledge bases for the cash-discipline video service. Everything runs on the local machine: the VLM through
Ollama, the embedder in-process on the GPU, vectors as files on disk.

Implemented: **empirical case memory** built from the 40 labeled training videos.
Regulatory retrieval over the NPA corpus is a separate project: `../rag`.

The training videos are not used as a class list. The test set may contain violation types that never appear in
them, so the case memory stores what was observed and how it relates to the label, and leaves the legal meaning
to the regulatory layer:

```
video -> clip windows -> embeddings ----------> similar training cases (normal vs violation, same kind of scene)
      -> VLM facts (states, actions) ---------> observable facts for rule verification
      -> labeled events -> description, type hypothesis, candidate rules (verified citations from cashwatch)
```

## Run

```powershell
uv sync                      # torch 2.8.0 + CUDA 12.8 (RTX 50xx), transformers 4.57.1
uv run kb cases labels       # spreadsheet -> labels.json (40 unique videos, 22 events, 12 camera setups)
uv run kb cases frames       # frames at 2 fps, 4 s windows with 2 s stride, 10 s clips around events
uv run kb cases annotate     # VLM passes through Ollama: events, overview, facts (resumable)
uv run kb cases cases        # assemble cases.jsonl + text cards
uv run kb cases embed        # Qwen3-VL-Embedding-2B vectors: clip, frame, card
uv run kb cases eval         # leave-one-setup-out evaluation -> eval.json
uv run pytest
```

`annotate` and `embed` never share the GPU: `embed` unloads every Ollama model first; Ollama loads the VLM again on
the next annotate call.

Settings (env vars, see `src/forte_kb/config.py`): `KB_VIDEOS_DIR`, `KB_LABELS_XLSX`, `KB_CASES_DIR`,
`KB_EMBED_MODEL` (local dir or HF repo id already in the local cache, default `Qwen/Qwen3-VL-Embedding-2B`, loaded offline), `KB_EMBED_DIM` (MRL, default 2048),
`KB_VLM_MODEL` (default `qwen3.8:27b`), `KB_OLLAMA_URL`, `KB_VLM_NUM_CTX`, `KB_CASHWATCH_REGISTRY`.

## Store layout (`store/cases/`)

| Path | Content |
|---|---|
| `labels.json` | unique videos (X-4 = X-47 and X-5 = X-48 merged by MD5), events with ids `X-4#e1`, camera setup group, ffprobe data |
| `frames/<video>/tNNNNNNN.jpg` | every video decoded once at 2 fps (name = milliseconds) |
| `windows.json` | 4 s windows, 2 s stride, with relation to the labels (see below) |
| `clips/<video>_eN.mp4` | 10 s around each labeled event, for human review |
| `vlm/{events,overview,facts}.jsonl` | raw VLM output with model, prompt version, frame times, latency |
| `cases.jsonl` | one case per video (schema below) |
| `cards.jsonl` | Russian text cards: one per event, one per video, one per annotated 4 s window |
| `vectors/{clip,frame,card}.npy` + `.jsonl` | L2-normalized embeddings and metadata per row |
| `review_template.json` | all events with the current type hypothesis and the VLM's type, to start a manual review |
| `eval.json` | leave-one-setup-out results |
| `flag_priors.json` | per facts flag: rate in event vs normal windows, number of violation vs clean videos with it |

Window relations: `event` (contains a labeled second), `near` (within 3 s of one), `other` (rest of a violation
video: unlabeled, a violation can persist), `normal` (clean video). Only `event` and `normal` are used as examples.

Case record (`cases.jsonl`):

```json
{
  "video_id": "X-24", "group": "s09", "ground_truth": {"label": "violation", "events": [{"event_id": "X-24#e1", "t": 8.0}]},
  "same_setup_videos": ["X-25", "X-26", "..."],
  "overview": {"layout_ru": "...", "people_ru": "...", "procedure_ru": "...", "objects": ["..."]},
  "events": [{
    "event_id": "X-24#e1", "t": 8.0,
    "clip": {"start": 3.0, "end": 13.0, "file": "clips/X-24_e1.mp4"},
    "keyframes": [{"t": 6.0, "file": "frames/X-24/t0006000.jpg"}],
    "state_changes": [{"t": 9.0, "concept": "cashier_leaving", "from": false, "to": true}],
    "actions": [{"t_start": 8.0, "t_end": 10.0, "actor": "cashier", "action": "walks_out_of_door", "object": "door"}],
    "vlm": {"before_ru": "...", "at_label_ru": "...", "after_ru": "...", "visual_events": [], "violation_type": "workplace_abandoned",
            "distinguishing_cue_ru": "...", "evidence_t": 11.0, "confidence": 0.95, "provenance": "label-conditioned VLM, unreviewed"},
    "type": {"id": "workplace_abandoned", "title": "...", "status": "likely", "source": "frame-review hypothesis", "vlm_agrees": true},
    "candidate_rules": [{"act": "nbrk-231", "clause": "п. 22", "quote": "..."}],
    "review": {"status": "unreviewed", "notes": ""}
  }],
  "timeline": [{"t": 0.5, "people": 1, "cashier_at_workplace": true, "safe_open": false, "...": "..."}],
  "actions": [], "scenes": [], "manual_frame_facts": []
}
```

Provenance of every field:

- `ground_truth`: organizers' spreadsheet only.
- `type`: hypothesis from frame review, overridden by `cases_review.json` once someone has watched the clip.
- `candidate_rules`: verified citations of that type from `cashwatch/src/cashwatch/rules/registry.json`; never
  generated by a model.
- `vlm`, `timeline`, `actions`, `scenes`, `overview`: local VLM output. The `events` pass is told where the label is,
  so it explains the label rather than detects it; treat it as a hypothesis.
- `manual_frame_facts`: hand-labelled frames of the VLM bake-off.

## Manual review

1. Copy `store/cases/review_template.json` to `cases_review.json` in this folder.
2. Watch `store/cases/clips/<event>.mp4`. Start with events where `type` and `vlm_type` disagree.
3. Set `type` (a registry type id or `new` with a `title`) and `type_status: "confirmed"`.
4. `uv run kb cases cases` and `uv run kb cases embed --spaces card`.

## Using the memory at inference time

```python
from forte_kb.cases.memory import CaseMemory
from forte_kb.embedder import CLIP, Embedder, Item

mem = CaseMemory("store/cases")
emb = Embedder(model_path)                      # load, embed the new video's windows, then emb.unload()
vecs = emb.encode([Item(frames=window_frames, instruction=CLIP) for window_frames in windows])
mem.suspiciousness(vecs)                        # share of violation windows among the nearest training windows
mem.similar_clips(vecs[i], k=5)                 # nearest cases with description, type and candidate rules
```

Text queries (for example a VLM observation of the new video) go through `similar_cards` with a vector from
`Item(text=..., instruction=TEXT_TO_CARD)`.

## Results (2026-10-17, leave-one-setup-out, 21 violation / 19 clean videos)

Each camera setup is held out in turn and scored against the other 11. Video AUC uses the best window per video.

| Signal | Window AUC | Video AUC | Best window within 2 s / 5 s |
|---|---|---|---|
| zero-shot qwen3.8:27b, whole video (Datathon baseline) | - | 0.54 | 5 / 5 of 15 flagged |
| clip embeddings, kNN | 0.55 | 0.46 | 4 / 8 of 21 |
| clip embeddings, centered per video, kNN | 0.60 | 0.63 | 5 / 11 |
| frame embeddings, probe | 0.57 | 0.50 | 9 / 12 |
| text cards (VLM scene lines), probe | 0.67 | 0.59 | 5 / 12 |
| **VLM facts, rule flags, no training** | **0.80** | **0.65** | 7 / 13 |
| VLM facts, probe | 0.63 | 0.61 | 8 / 14 |

What this means:

- Generic visual embeddings encode the room and the people, not the action: they do not separate violation
  from normal across setups. Use them to fetch similar cases as context, not to decide.
- Observed facts separate windows well (0.80), but single facts are common in normal work:
  `flag_priors.json` shows a phone in hand in 12 of 19 clean videos, the door opening in 13, the cashier leaving
  in 10. Combinations (leaving with cash on the desk, three or more people, cash unattended) never occur in clean
  videos but are rare. Context decides, which is what the VLM verification step with retrieved normal and
  violation cases has to supply.
- The nearest event case from another setup has the same type hypothesis in 42 % of event windows
  (majority type: 25 %).
- With 40 videos the 95 % interval of a video AUC is roughly ±0.18; differences below that are not evidence.

## Limitations

- 22 labeled events over 12 camera setups: retrieval numbers have wide confidence intervals; the evaluation holds
  out whole setups so that near-copies from the same room never leak.
- Labels give the second, not the onset; windows around the label (`near`) are excluded from training and scoring.
- Type hypotheses are unconfirmed until reviewed; the VLM disagrees with some of them (for example X-20).
- Records loaded into Pinecone Local do not persist; the files in `store/` are the source of truth.
