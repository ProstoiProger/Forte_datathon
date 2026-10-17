"""Retrieval evaluation against verified citations.

Gold: the cashwatch rule registry, where every violation type has citations (act, clause, verbatim quote) checked
against adilet.zan.kz. A chunk is relevant to a citation when it contains the quote. Two query sets:
  types         type title + rationale (written by someone who read the norms: optimistic for word overlap)
  observations  what the VLM described at a labeled training event (case memory, read-only), for events whose
                type the VLM and the hypothesis agree on, or that a reviewer confirmed (realistic runtime query)
Metrics over queries: hit@1/5/10 (a relevant chunk in the top k), MRR@10, and citation recall@10 (share of the
type's citations with a relevant chunk in the top 10).
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from .corpus import Chunk
from .hybrid import Retriever, rerank_text, rrf_fuse

_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "ё": "е", "–": "-", "—": "-"})


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_QUOTES)).strip(" .;:")


def gold_citations(registry_path: Path, chunks: list[Chunk]) -> tuple[dict, list[dict]]:
    """type id -> {title, rationale, citations: [{act_id, clause, chunk_ids}]}, plus citations not found."""
    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    act_ids = {k: v["url"].rstrip("/").split("/")[-1] for k, v in reg["acts"].items()}
    by_act: dict[str, list[tuple[int, str]]] = {}
    for i, c in enumerate(chunks):
        by_act.setdefault(c.act_id, []).append((i, norm(c.text)))
    types, missing = {}, []
    for tid, t in reg["types"].items():
        cits = []
        for cit in t.get("citations", []):
            if not cit.get("verified"):
                continue
            act = act_ids.get(cit["act"])
            q = norm(cit["quote"])
            probe = q[:120]  # a quote can run past a chunk split; its beginning identifies the unit
            found = [i for i, txt in by_act.get(act, []) if probe in txt]
            entry = {"act": cit["act"], "act_id": act, "clause": cit["clause"], "chunks": found}
            (cits if found else missing).append(entry)
            if not found:
                entry["type"] = tid
        if cits:
            types[tid] = {"title": t["title"], "rationale": t["rationale"], "citations": cits}
    return types, missing


def query_sets(types: dict, cases_path: Path) -> dict[str, list[dict]]:
    sets = {"types": [{"id": tid, "type": tid, "query": f"{t['title']}. {t['rationale']}"} for tid, t in types.items()]}
    obs = []
    if cases_path.exists():
        for line in cases_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            for e in json.loads(line)["events"]:
                vlm, typ = e.get("vlm") or {}, e["type"]["id"]
                trusted = e["type"]["status"] == "confirmed" or vlm.get("violation_type") == typ
                if vlm and trusted and typ in types:
                    obs.append({"id": e["event_id"], "type": typ,
                                "query": f"{vlm.get('at_label_ru', '')} {vlm.get('distinguishing_cue_ru', '')}".strip()})
    sets["observations"] = obs
    return sets


def _metrics(ranked: list[int], cits: list[dict], k: int = 10) -> dict:
    gold = {i for c in cits for i in c["chunks"]}
    pos = next((r for r, i in enumerate(ranked[:k], 1) if i in gold), None)
    top = set(ranked[:k])
    return {"hit1": pos == 1, "hit5": pos is not None and pos <= 5, "hit10": pos is not None,
            "mrr": 1 / pos if pos else 0.0,
            "cit_recall": sum(bool(set(c["chunks"]) & top) for c in cits) / len(cits)}


def evaluate(retriever: Retriever, types: dict, sets: dict[str, list[dict]], rerank: bool) -> dict:
    mask = retriever.mask()
    configs = ["bm25", "dense", "rrf"] + (["rrf+rerank"] if rerank else [])
    out = {}
    for name, queries in sets.items():
        rows = {c: [] for c in configs}
        lat = {c: [] for c in ("bm25", "embed_query", "dense", "rerank")}
        per_query = []
        for q in queries:
            timings: dict[str, float] = {}
            lists = retriever.candidates(q["query"], mask, timings=timings)
            for key, v in timings.items():
                lat[key].append(v)
            fused = [i for i, _, _ in rrf_fuse(lists)]
            ranked = {"bm25": lists["bm25"], "dense": lists["dense"], "rrf": fused}
            if rerank:
                t = time.perf_counter()
                depth = fused[:30]
                scores = retriever.reranker.score(q["query"], [rerank_text(retriever.chunks[i]) for i in depth])
                lat["rerank"].append((time.perf_counter() - t) * 1000)
                ranked["rrf+rerank"] = [depth[j] for j in np.argsort(-scores, kind="stable")] + fused[30:]
            cits = types[q["type"]]["citations"]
            row = {"id": q["id"], "type": q["type"]}
            for c in configs:
                m = _metrics(ranked[c], cits)
                rows[c].append(m)
                row[c] = round(m["mrr"], 3)
            per_query.append(row)
        out[name] = {
            "queries": len(queries),
            "metrics": {c: {k: round(float(np.mean([r[k] for r in rows[c]])), 3) for k in rows[c][0]}
                        for c in configs} if queries else {},
            "latency_ms_median": {k: round(float(np.median(v)), 1) for k, v in lat.items() if v},
            "per_query_mrr": per_query,
        }
    return out
