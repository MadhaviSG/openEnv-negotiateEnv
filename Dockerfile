# NegotiateEnv — HuggingFace Spaces Deployment
# This Dockerfile is optimized for HF Spaces (port 7860 required)

FROM python:3.11-slim

WORKDIR /app

# Copy all project files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')"

# Start the FastAPI server
CMD ["uvicorn", "negotiate_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
