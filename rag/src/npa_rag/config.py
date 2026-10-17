"""Paths and model settings. Every value can be overridden with an environment variable of the same name."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FORTE = PROJECT.parent

NPA_DIR = Path(os.environ.get("RAG_NPA_DIR", str(FORTE / "npa_bank_kz")))
STORE = Path(os.environ.get("RAG_STORE", str(PROJECT / "store")))
CASHWATCH_REGISTRY = Path(os.environ.get(
    "RAG_CASHWATCH_REGISTRY", str(FORTE / "cashwatch" / "src" / "cashwatch" / "rules" / "registry.json")))
CASES_JSONL = Path(os.environ.get("RAG_CASES_JSONL", str(FORTE / "knowledge" / "store" / "cases" / "cases.jsonl")))

# Acts indexed by default; amending acts are left out (their changes are already in the consolidated texts).
INDEX_KINDS = tuple(os.environ.get("RAG_INDEX_KINDS", "base,repealed").split(","))
# Acts returned by default; repealed acts stay searchable only on request.
SEARCH_KINDS = tuple(os.environ.get("RAG_SEARCH_KINDS", "base").split(","))

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "2048"))
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Instructions in English, as the Qwen3-VL-Embedding model card recommends.
DOC_INSTRUCTION = "Represent this provision of Kazakhstan banking and cash-handling regulations for retrieval."
QUERY_INSTRUCTION = ("Given a description of a situation or a violation in a bank cash room, retrieve the provisions "
                     "of Kazakhstan regulations that apply to it.")

RRF_K = 60
CANDIDATES = 100      # per retriever, before fusion
RERANK_DEPTH = 30     # fused candidates passed to the cross-encoder
