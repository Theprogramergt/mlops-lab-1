# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only dependency files first, so this layer is cached
# unless pyproject.toml or uv.lock actually change
COPY pyproject.toml uv.lock ./

# Install dependencies only, without the project itself — this layer
# stays cached even when your source code changes
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the source code (changes here won't invalidate the layer above)
COPY src/ ./src/
COPY README.md ./README.md

# Install the project itself, now that src/ is present
RUN uv sync --frozen --no-dev

# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Bring in only the built virtual environment, not uv or build tools
COPY --from=builder /app/.venv /app/.venv

# Copy the source code into the runtime image too
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]