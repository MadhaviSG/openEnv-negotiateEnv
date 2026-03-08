# NegotiateEnv — OpenEnv-compatible B2B SaaS negotiation environment
# Exposes port 7860 for HuggingFace Spaces.

FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app/env

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY negotiate_env/ negotiate_env/

# Install dependencies into a venv using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Final runtime stage
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 7860

CMD ["uvicorn", "negotiate_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]