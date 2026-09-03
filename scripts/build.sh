#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --frozen python -m unittest discover -s tests
docker build -t neoprego-audio2026:latest .
