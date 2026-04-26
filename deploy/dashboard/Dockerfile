# Dockerfile for the TokenEfficiencyEnv Playground (web_dashboard.py).
#
# This is a SEPARATE deployable from the OpenEnv environment server in
# token_efficiency_env/server/Dockerfile. It ships the human-facing,
# light-theme dashboard with structured Budget/Answer inputs.
#
# Why a slim base instead of openenv-base:
#   * The dashboard does not need OpenEnv's HTTP server runtime — it
#     imports TokenEfficiencyEnvironment in-process and exposes its own
#     FastAPI surface. A slim image gives faster cold-starts on free
#     HF Space hardware (cpu-basic) where image pulls dominate boot time.
#   * Dependencies are resolved from a flat requirements.txt that pins the
#     direct deps the dashboard actually imports (no uv lockfile drift).
#
# Hugging Face Spaces convention:
#   * The Space sets ``$PORT`` in the runtime environment (typically 7860).
#   * ``app_port: 7860`` in the SPACE README frontmatter must match the
#     EXPOSE / CMD below.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DASHBOARD_PORT=7860

WORKDIR /app

# System deps: curl is used by the HEALTHCHECK; build-essential is needed
# transiently by tokenizers' Rust extension on some platforms.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Layer 1 — dependencies. Copying requirements.txt first lets Docker cache
# this layer across code-only edits, which dramatically speeds up
# Space rebuilds while iterating on dashboard.html or the dashboard router.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Layer 2 — application code.
# The push script stages this Dockerfile alongside web_dashboard.py,
# dashboard.html, and the token_efficiency_env/ package, so a flat COPY
# is enough.
COPY . /app

# HF Spaces routes outside traffic to $PORT; we default to 7860 to match
# the SPACE README frontmatter. Override locally with -e DASHBOARD_PORT=...
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${DASHBOARD_PORT:-7860}/api/state" >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn web_dashboard:app --host 0.0.0.0 --port ${DASHBOARD_PORT:-7860}"]
