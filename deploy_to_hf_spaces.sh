#!/bin/bash
# Deploy NegotiateEnv to HuggingFace Spaces
# Usage: ./deploy_to_hf_spaces.sh YOUR_USERNAME negotiate-env

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hf_username> <space_name>"
    echo "Example: $0 mayukareddy negotiate-env"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "=================================================="
echo "Deploying NegotiateEnv to HuggingFace Spaces"
echo "=================================================="
echo "Username: ${HF_USERNAME}"
echo "Space: ${SPACE_NAME}"
echo "URL: ${SPACE_URL}"
echo ""

# Check if space exists (clone or create)
if git ls-remote "https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}" &>/dev/null; then
    echo "✅ Space exists. Cloning..."
    git clone "https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}" hf_space_temp
else
    echo "⚠️  Space doesn't exist yet."
    echo "   Please create it first at: https://huggingface.co/new-space"
    echo "   - Name: ${SPACE_NAME}"
    echo "   - SDK: Docker"
    echo "   - Hardware: CPU Basic (free)"
    echo ""
    read -p "Press Enter after creating the space..."
    git clone "https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}" hf_space_temp
fi

cd hf_space_temp

# Copy necessary files
echo ""
echo "📦 Copying files..."
cp -r ../negotiate_env .
cp ../Dockerfile .
cp ../requirements.txt .
cp ../pyproject.toml .
cp ../.dockerignore .

# Create a README for the Space
cat > README.md << 'EOF'
---
title: NegotiateEnv
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# NegotiateEnv — OpenEnv B2B SaaS Negotiation Environment

An OpenEnv-compatible RL environment where LLM agents learn to negotiate enterprise SaaS contracts.

## API Endpoints

- `GET /health` - Health check
- `POST /reset` - Reset environment and get initial observation
- `POST /step` - Take an action and get next observation
- `GET /state` - Get current environment state

## Example Usage

```python
import requests

# Reset environment
response = requests.post("https://YOUR_USERNAME-negotiate-env.hf.space/reset", json={})
obs = response.json()

# Take action
action = {
    "action_type": "counter",
    "price_per_seat": 150.0,
    "contract_length": 2.0,
    "annual_increase_cap": 5.0,
    "message": "Can we do $150/seat for 2 years?"
}
response = requests.post("https://YOUR_USERNAME-negotiate-env.hf.space/step", json=action)
obs = response.json()
```

## Repository

https://github.com/YOUR_USERNAME/negotiate-env

## Hackathon

Built for OpenEnv Hackathon SF 2024
EOF

# Commit and push
echo ""
echo "📤 Pushing to HuggingFace..."
git add .
git commit -m "Deploy NegotiateEnv environment server"
git push

cd ..
rm -rf hf_space_temp

echo ""
echo "=================================================="
echo "✅ Deployment complete!"
echo "=================================================="
echo ""
echo "Your environment will be available at:"
echo "  ${SPACE_URL}"
echo ""
echo "Wait 2-3 minutes for the build to complete, then test:"
echo "  curl https://${HF_USERNAME}-${SPACE_NAME}.hf.space/health"
echo ""
echo "Use this URL in your Colab training notebook:"
echo "  ENV_URL = \"https://${HF_USERNAME}-${SPACE_NAME}.hf.space\""
echo ""
