# Regulation search service: BM25 + dense vectors + reciprocal rank fusion, embedder on CPU (the GPU is for the VLM).
# Build context is the repository root, because rag uses the embedder wrapper from ../knowledge.
# The indexes (rag/store) and the Hugging Face model cache are mounted read-only; loading never goes online.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    RAG_STORE=/data/store

# CPU build of the pinned torch: the projects' uv config points at the CUDA index for the RTX 5090 host
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY knowledge/pyproject.toml knowledge/README.md knowledge/
COPY knowledge/src knowledge/src
RUN pip install ./knowledge
COPY rag/pyproject.toml rag/README.md rag/
COPY rag/src rag/src
RUN pip install ./rag && python -c "import npa_rag.server, forte_kb.embedder"

EXPOSE 8765
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if b'\"ready\":true' in urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4).read() else 1)"
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8765", "--device", "cpu"]
