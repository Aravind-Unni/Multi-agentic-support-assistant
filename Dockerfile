# Uses Astral's official uv image, which ships uv preinstalled on top of
# a slim Python base. Check your .python-version file and match the tag
# below if it's not 3.12 (e.g. python3.11-bookworm-slim).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# --- Application code ---
COPY src ./src
COPY frontend ./frontend


RUN uv sync --frozen --no-dev



EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
