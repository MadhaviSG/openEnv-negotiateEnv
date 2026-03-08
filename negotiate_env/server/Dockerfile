# NegotiateEnv — OpenEnv-compatible B2B SaaS negotiation environment
# Exposes port 7860 for HuggingFace Spaces.
# Build from repo root: docker build -f negotiate_env/server/Dockerfile .

FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir openenv-core>=0.2.1 fastapi uvicorn pydantic && \
    pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "negotiate_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
