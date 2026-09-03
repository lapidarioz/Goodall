FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ARG UV_VERSION=0.10.2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY neoprego_audio2026 ./neoprego_audio2026
COPY models ./models
RUN uv sync --frozen --no-dev

COPY scripts/download_panns_checkpoint.py ./scripts/download_panns_checkpoint.py

ENTRYPOINT ["uv", "run", "neoprego-audio2026", "--dataset-root", "/data"]
