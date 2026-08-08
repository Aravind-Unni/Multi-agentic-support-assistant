 FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# --- Application code ---
COPY src ./src
COPY frontend ./frontend
COPY data ./data

# Install the project itself now that source is present
RUN uv sync --frozen --no-dev


RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# not generated output — the vector index is built in memory at every
# app startup (see RAG_setup.py), so nothing here needs a persistent
# volume.

EXPOSE 8000


CMD uv run uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}