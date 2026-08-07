# Uses Astral's official uv image, which ships uv preinstalled on top of
# a slim Python base. Check your .python-version file and match the tag
# below if it's not 3.12 (e.g. python3.11-bookworm-slim).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# --- Dependency layer (cached separately from source code) ---
# Copying only the lock/manifest first means Docker can reuse this layer
# on rebuilds as long as dependencies haven't changed, even if src/ has.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# --- Application code ---
COPY src ./src
COPY frontend ./frontend
COPY data ./data

# Install the project itself now that source is present
RUN uv sync --frozen --no-dev

# data/ contains static source input (orders.json, trendly_policy.md),
# not generated output — the vector index is built in memory at every
# app startup (see RAG_setup.py), so nothing here needs a persistent
# volume.

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]