# Deploy NegotiateEnv to HuggingFace Spaces

This guide shows how to deploy the NegotiateEnv server to HuggingFace Spaces as a Docker container.

## Prerequisites

1. HuggingFace account: https://huggingface.co/join
2. Git installed locally
3. Docker installed (for local testing)

## Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `negotiate-env` (or your preferred name)
   - **License**: MIT
   - **Select the Space SDK**: Docker
   - **Space hardware**: CPU Basic (free) - sufficient for the environment server
   - **Visibility**: Public

3. Click "Create Space"

## Step 2: Prepare Files for HF Spaces

HuggingFace Spaces expects specific files in the root directory. We need to create:

```
.
├── Dockerfile          # Points to our actual Dockerfile
├── README.md          # Space description (auto-generated)
├── requirements.txt   # Python dependencies
└── negotiate_env/     # Your package
```

## Step 3: Clone Your Space Repository

```bash
# Clone the space (replace YOUR_USERNAME and negotiate-env with your values)
git clone https://huggingface.co/spaces/YOUR_USERNAME/negotiate-env
cd negotiate-env

# Copy your negotiate_env package
cp -r /path/to/your/negotiate-env/negotiate_env .
cp /path/to/your/negotiate-env/pyproject.toml .
cp /path/to/your/negotiate-env/requirements.txt .
```

## Step 4: Create Root Dockerfile

Create a `Dockerfile` in the root that uses our existing one:

```dockerfile
# Use our existing Dockerfile
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Expose port 7860 (HF Spaces requirement)
EXPOSE 7860

# Start the server
CMD ["uvicorn", "negotiate_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Step 5: Create .gitignore

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.env
.venv
venv/
*.log
.DS_Store
EOF
```

## Step 6: Push to HuggingFace

```bash
git add .
git commit -m "Initial deployment of NegotiateEnv"
git push
```

## Step 7: Wait for Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/negotiate-env`
2. Watch the build logs in the "Logs" tab
3. Wait for "Running on http://0.0.0.0:7860" message
4. Your space will show "Running" status

## Step 8: Test Your Deployed Environment

```bash
# Test health endpoint
curl https://YOUR_USERNAME-negotiate-env.hf.space/health

# Test reset endpoint
curl -X POST https://YOUR_USERNAME-negotiate-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Step 9: Use in Training

Your environment is now accessible at:
```
https://YOUR_USERNAME-negotiate-env.hf.space
```

Use this URL in your Colab training notebook!

## Troubleshooting

### Build fails with "No module named 'negotiate_env'"
- Make sure you copied the entire `negotiate_env/` directory
- Check that `pyproject.toml` is in the root

### Port 7860 not accessible
- HF Spaces requires port 7860 specifically
- Check your Dockerfile exposes and uses port 7860

### Server starts but endpoints return 404
- Check the logs for startup errors
- Verify FastAPI app is created correctly in `negotiate_env/server/app.py`

### Out of memory
- The free CPU Basic tier should be sufficient
- If needed, upgrade to CPU Upgrade ($0.03/hour)

## Environment Variables (Optional)

To use the HuggingFace dataset or change difficulty:

1. Go to your Space settings
2. Add environment variables:
   - `NEGOTIATE_DIFFICULTY`: `easy`, `medium`, or `hard`
   - `NEGOTIATE_USE_HF_DATASET`: `true` or `false`

## Next Steps

Once deployed:
1. ✅ Copy your Space URL
2. ✅ Use it in the Colab training notebook
3. ✅ Train your model
4. ✅ Submit to hackathon with your live Space URL
