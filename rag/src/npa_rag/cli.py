"""Command line: uv run rag <command> ..."""

import argparse
import json
import logging
import os
import sys
import time

from . import config


def _embedder(device: str | None):
    from forte_kb.embedder import Embedder
    return Embedder(config.EMBED_MODEL, config.EMBED_DIM, device=device)


def _free_gpu() -> None:
    """The embedder and the large VLM never share the GPU: evict Ollama models before loading the embedder."""
    try:
        from forte_kb.vlm import Ollama
        freed = Ollama("http://127.0.0.1:11434", "").unload_all()
        if freed:
            logging.info("unloaded from Ollama: %s", ", ".join(freed))
    except Exception as exc:
        logging.info("ollama not reachable (%s); continuing", exc)


def _serve(host: str, port: int, device: str, rerank: bool) -> int:
    import uvicorn

    from .server import create_app
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # torch never opens a CUDA context next to the VLM
    uvicorn.run(create_app(config.STORE, device=device, rerank=rerank), host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "transformers", "bm25s"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build", help="chunks + BM25 + dense vectors")
    s.add_argument("--skip-dense", action="store_true")
    s.add_argument("--kinds", default=",".join(config.INDEX_KINDS))
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--limit", type=int, default=0, help="index only the first N chunks (smoke test)")
    s = sub.add_parser("search", help="hybrid search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--kinds", default=",".join(config.SEARCH_KINDS))
    s.add_argument("--methods", default="bm25,dense")
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--device", default=None, help="cuda or cpu for the embedder and the reranker")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("eval", help="retrieval quality against the verified citations of the rule registry")
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--device", default=None)
    s = sub.add_parser("serve", help="HTTP search service (GET /health, POST /search)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=47865)
    s.add_argument("--device", default="cpu", help="cpu (default: the GPU belongs to the VLM) or cuda")
    s.add_argument("--rerank", action="store_true", help="load the cross-encoder (about 15 s per query on CPU)")
    args = p.parse_args(argv)

    store = config.STORE
    if args.cmd == "serve":
        return _serve(args.host, args.port, args.device, args.rerank)
    if args.cmd == "build":
        from .corpus import build_chunks, save_chunks
        from .sparse import Sparse
        t = time.time()
        chunks = build_chunks(config.NPA_DIR, set(args.kinds.split(",")))
        if args.limit:
            chunks = chunks[: args.limit]
        save_chunks(chunks, store / "chunks.jsonl")
        logging.info("chunks: %d in %.1f s", len(chunks), time.time() - t)
        t = time.time()
        Sparse.build([c.index_text() for c in chunks], [c.id for c in chunks]).save(store / "bm25")
        logging.info("bm25 built in %.1f s", time.time() - t)
        if not args.skip_dense:
            from .dense import build
            _free_gpu()
            emb = _embedder(None)
            t = time.time()
            try:
                build(chunks, emb, store, args.batch_size)
            finally:
                emb.unload()
            logging.info("dense built in %.1f s", time.time() - t)
        return 0

    from .hybrid import Retriever
    methods = tuple(getattr(args, "methods", "bm25,dense").split(","))
    emb = _embedder(args.device) if "dense" in methods else None
    rr = None
    if args.rerank:
        from .rerank import Reranker
        rr = Reranker(config.RERANK_MODEL, device=args.device)
    retriever = Retriever(store, embedder=emb, reranker=rr)

    if args.cmd == "search":
        res = retriever.search(args.query, k=args.k, kinds=tuple(args.kinds.split(",")), methods=methods,
                               rerank=args.rerank)
        if args.json:
            print(json.dumps({"query": res.query, "timings_ms": res.timings_ms,
                              "hits": [h.as_dict() for h in res.hits]}, ensure_ascii=False, indent=1))
            return 0
        for n, h in enumerate(res.hits, 1):
            c = h.chunk
            rr_s = f" rerank={h.rerank:.3f}" if h.rerank is not None else ""
            print(f"{n}. [{c.status}] {c.act_requisites}\n   {c.citation}{' (' + c.part + ')' if c.part else ''}")
            print(f"   ranks={h.ranks} rrf={h.rrf:.4f}{rr_s}\n   {c.text[:300].replace(chr(10), ' / ')}\n")
        print("timings ms:", {k: round(v, 1) for k, v in res.timings_ms.items()})
        return 0

    if args.cmd == "eval":
        from .evaluate import evaluate, gold_citations, query_sets
        types, missing = gold_citations(config.CASHWATCH_REGISTRY, retriever.chunks)
        if missing:
            logging.warning("citations without a matching chunk: %s",
                            [(m["type"], m["act"], m["clause"]) for m in missing])
        sets = query_sets(types, config.CASES_JSONL)
        res = evaluate(retriever, types, sets, args.rerank)
        res["gold"] = {"types": len(types), "citations": sum(len(t["citations"]) for t in types.values()),
                       "missing": [(m["type"], m["act"], m["clause"]) for m in missing]}
        (store / ("eval_rerank.json" if args.rerank else "eval.json")).write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        for name in sets:
            r = res[name]
            print(f"\n{name}: {r['queries']} queries; median latency ms {r['latency_ms_median']}")
            for cfg, m in r["metrics"].items():
                print(f"  {cfg:11} " + "  ".join(f"{k}={v}" for k, v in m.items()))
        print("\ngold:", res["gold"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
