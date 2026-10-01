# LawCast semantic search sidecar (FastAPI + KURE-v1 + FAISS).
#
# Serving-only image: index artifacts are bind-mounted from the host at runtime
# (see docker-compose.yml / README "아티팩트 교체 규칙") and the HuggingFace
# model cache lives in a named volume, so the image itself stays code-only and
# rebuilds are cheap. The healthcheck is owned by docker-compose.yml, matching
# the sibling services.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Seoul \
    HF_HOME=/cache/huggingface \
    LAWCAST_SEMANTIC_ARTIFACTS_DIR=/app/artifacts

# Timezone data for TZ=Asia/Seoul log timestamps (same convention as backend).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && echo "Asia/Seoul" > /etc/timezone

WORKDIR /app

# CPU-only torch first: the default PyPI linux wheel pulls multi-GB CUDA
# dependencies this CPU inference sidecar never uses. The version matches
# requirements.lock; `torch==X` accepts the `X+cpu` local wheel version.
ARG TORCH_VERSION=2.14.0
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

# Pinned, test-verified dependency set (torch is already satisfied above, so
# pip keeps the CPU wheel).
COPY requirements.lock ./
RUN pip install -r requirements.lock

# Non-root runtime user, fixed at uid/gid 1001 to match the compose `user:`
# override and the backend data-file uid: the named HF cache volume inherits
# this ownership on copy-on-first-use, so the serving process can download and
# refresh the KURE-v1 weights into its own cache without any manual chown.
# The artifacts bind mount is host-owned and read-mostly.
RUN addgroup --gid 1001 semantic \
    && adduser --uid 1001 --ingroup semantic --disabled-password --gecos "" \
        --no-create-home semantic \
    && mkdir -p /cache/huggingface /app/artifacts \
    && chown -R semantic:semantic /cache /app

COPY --chown=semantic:semantic lawcast_semantic/ lawcast_semantic/
COPY --chown=semantic:semantic service/ service/
# scripts/ ships for operational runs via `docker compose exec` (e.g. the
# incremental index update); the image never runs the offline pipeline itself.
COPY --chown=semantic:semantic scripts/ scripts/

USER semantic

EXPOSE 8300

CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8300"]
