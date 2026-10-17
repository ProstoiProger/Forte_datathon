# rag

Hybrid retrieval over the Kazakhstan banking cash-desk regulations in `../npa_bank_kz` (313 acts from
adilet.zan.kz, state of the database 14.10.2026). Fully local: BM25 and vector search run in-process, the
embedder and the optional cross-encoder load from the local Hugging Face cache and never go online.

```
query text ─┬─► BM25 (Russian Snowball stemming + stopwords) ── top 100 ─┐
            └─► Qwen3-VL-Embedding-2B (text) · cosine ───────── top 100 ─┴─► RRF (k=60) ─► [bge-reranker-v2-m3] ─► top k
```

Reciprocal rank fusion scores a chunk by `Σ 1 / (60 + rank)` over the retrievers, so BM25 scores and cosine
similarities never need to be calibrated against each other. The cross-encoder rescores the top 30 fused chunks.

## Run

```powershell
uv sync
uv run rag build                       # chunks + BM25 (seconds) + dense vectors (GPU; frees Ollama first)
uv run rag search "кассир ушёл, оставив наличные на столе" -k 5
uv run rag search "..." --rerank       # add the cross-encoder
uv run rag search "..." --methods bm25 # one retriever only; --json for machine output
uv run rag eval                        # quality against verified citations; --rerank to include the reranker
uv run pytest
```

Settings (env vars, `src/npa_rag/config.py`): `RAG_NPA_DIR`, `RAG_STORE`, `RAG_INDEX_KINDS` (default
`base,repealed`), `RAG_SEARCH_KINDS` (default `base`), `RAG_EMBED_MODEL`, `RAG_EMBED_DIM`, `RAG_RERANK_MODEL`.

The embedder wrapper and the vector store come from `../knowledge` (`forte_kb.embedder`, `forte_kb.store`),
installed read-only as a path dependency; nothing in `../knowledge` is modified.

## Chunks

One chunk = one legal unit with an exact citation: a point (`п. 75`) of a resolution, rules or appendix, or an
article (`ст. 252-1`) of a code or law, with its heading path (part, chapter, paragraph) and the document it
belongs to (the resolution itself, the rules it approves, or an appendix). Units longer than 1800 characters are
split at line boundaries and keep the label (`part: "2/3"`). Footnotes about amendments ("Сноска.") and editorial
notes ("Примечание ИЗПИ!") are stored in `notes` and not indexed.

| Acts | Indexed | Returned by default | Why |
|---|---|---|---|
| base (current consolidated texts) | yes | yes | the norms to cite |
| repealed ("Утративший силу") | yes | no (`--kinds base,repealed`) | history only; citing them as current would be wrong |
| amending acts (91_amendments) | no | - | their changes are already merged into the consolidated texts |

Each hit carries `citation`, `act_requisites`, `label`, `status`, `source` (adilet URL), the verbatim `text`,
its rank in each retriever and the RRF / reranker scores, so the link from a result to the official text is
explicit.

## Store (`store/`)

| Path | Content |
|---|---|
| `chunks.jsonl` | all chunks with metadata |
| `bm25/` | bm25s index + chunk ids |
| `dense.npy`, `dense.jsonl` | embeddings (rows in chunk order) + ids |
| `eval.json`, `eval_rerank.json` | evaluation results |

## Evaluation

Gold: the 32 verified citations (act, clause, verbatim quote) of the 15 violation types in the cashwatch rule
registry; a chunk is relevant when it contains the quote. Query sets: `types` (type title + rationale) and
`observations` (the VLM's description of labeled training events from the case memory, for events whose type is
confirmed or agreed by the VLM). See the Results section for the numbers.

## Results (2026-10-18, 26 182 chunks: 22 672 current + 3 510 repealed; search over current acts)

Observations use the 18 events whose type a reviewer confirmed (`knowledge/cases_review.json`, 2026-10-18) and
that map to a registry type; the 4 events of new types have no gold citation yet.

| Retrieval | Types (15): hit@1 / hit@5 / hit@10 / MRR | Observations (18): hit@1 / hit@5 / hit@10 / MRR |
|---|---|---|
| BM25 | 0.27 / 0.67 / 0.87 / 0.43 | 0.17 / 0.39 / 0.50 / 0.27 |
| dense | 0.27 / 0.60 / 0.87 / 0.44 | 0.11 / 0.44 / 0.61 / 0.25 |
| **RRF** | **0.40 / 0.87 / 0.93 / 0.61** | **0.22 / 0.72 / 0.78 / 0.42** |
| RRF + reranker | 0.47 / 0.80 / 0.93 / 0.63 | 0.22 / 0.56 / 0.61 / 0.38 |

One query is 7 points (types) or 6 points (observations): differences of one query are noise.

- RRF beats both retrievers on both sets (MRR 0.61 vs 0.43 / 0.44 and 0.42 vs 0.27 / 0.25): the two retrievers
  fail on different queries, and fusion keeps a norm that either of them ranks well.
- On the VLM's descriptions each retriever alone finds the norm in the top 10 for only half to 60 % of the
  events: the VLM says "встаёт и выходит через дверь", the norm says "покидать рабочее место".
- The reranker raises hit@1 by one type query, and on the observations it pushes correct norms out of the top 5
  (0.72 -> 0.56) and top 10 (0.78 -> 0.61). It is off by default.
- An earlier run on 11 observation queries (only events where the VLM agreed with the type hypothesis) looked
  better for the reranker; the confirmed, larger set reversed that.
- Not found by any method: phone use, food and drink, personal items. Their registry citation (Rules No. 231,
  item 12, the exhaustive list of what may be brought into the cash unit) is an interpretation; the text never
  mentions a phone or a drink. Such links have to stay in the human-verified registry.

Latency per query, median: BM25 0.5-15 ms, dense search 3 ms, query embedding 34 ms on GPU / 0.44 s on CPU,
reranking 30 chunks 0.26 s on GPU / 15 s on CPU. Building: chunks 1 s, BM25 2 s, dense vectors 17 min on the
RTX 5090 Laptop.
