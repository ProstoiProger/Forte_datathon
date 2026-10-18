# cashwatch: API, web UI and the facts analyzer. The VLM runs in Ollama (on the host or in the optional `ollama`
# compose service); the case base bundle and job data are mounted, nothing from the dataset is baked into the image.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN pip install "uv==0.12.7"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev && /opt/venv/bin/python -c "import cv2, cashwatch.main"

ENV PATH=/opt/venv/bin:$PATH \
    CASHWATCH_HOST=0.0.0.0 \
    CASHWATCH_PORT=8000 \
    CASHWATCH_DATA_DIR=/data/var \
    CASHWATCH_CASEBASE_DIR=/data/casebase

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if b'\"ready\":true' in urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).read() else 1)"
CMD ["cashwatch"]
