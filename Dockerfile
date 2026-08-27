# One image, one origin: the API serves the built UI at "/" and its own routes
# under "/api". That removes CORS from production and means Cloud Run runs a
# single service rather than two that have to find each other.
#
# Built by Cloud Build via `gcloud run deploy --source .`, so it never needs
# Docker installed locally.

# --- stage 1: build the frontend --------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Manifests first: this layer is cached unless dependencies actually change,
# which is most of the build time on a re-deploy.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# `tsc -b && vite build` - the typecheck is part of the build on purpose, so a
# type error fails here rather than shipping.
RUN npm run build


# --- stage 2: runtime --------------------------------------------------------
# Node does not survive into this stage; only the built assets do.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies before source, same caching reason as above. `--frozen` makes the
# build fail rather than silently resolve differently from uv.lock.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend/ ./
RUN uv sync --frozen --no-dev

COPY --from=frontend /build/dist ./static

ENV ONEMIND_STATIC_DIR=/app/static \
    ONEMIND_LLM_PROVIDER=groq \
    PORT=8080

# `tools/store.py` locates fixtures relative to its own file, which resolves
# correctly only because uv installs the project editable. That is uv's default
# rather than something this file controls, so assert it at build time - a
# broken layout should fail the build, not the demo.
RUN uv run python -c "from onemind.tools import store; assert store.patients(), 'fixtures did not load'"

EXPOSE 8080

# Cloud Run injects PORT and may not use 8080; shell form so it expands.
CMD uv run uvicorn onemind.api.main:app --host 0.0.0.0 --port ${PORT}
