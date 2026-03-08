#!/bin/bash
# Push to HuggingFace Spaces with authentication

echo "=================================================="
echo "Push NegotiateEnv to HuggingFace Spaces"
echo "=================================================="
echo ""
echo "You need a HuggingFace Access Token to push."
echo ""
echo "Get your token here:"
echo "https://huggingface.co/settings/tokens"
echo ""
echo "Steps:"
echo "1. Click 'New token'"
echo "2. Name it: 'negotiate-env-deploy'"
echo "3. Type: 'Write'"
echo "4. Click 'Generate'"
echo "5. Copy the token"
echo ""
read -p "Press Enter when you have your token ready..."
echo ""
echo "Now pushing to HuggingFace..."
echo ""

cd hf_space

# Configure git to use the token
git remote set-url origin https://KushalAdhyaru:$(read -sp "Paste your HuggingFace token: " token && echo $token)@huggingface.co/spaces/KushalAdhyaru/negotiate-env

echo ""
echo "Pushing..."
git push

echo ""
echo "=================================================="
echo "✅ Push complete!"
echo "=================================================="
echo ""
echo "Check your Space at:"
echo "https://huggingface.co/spaces/KushalAdhyaru/negotiate-env"
echo ""
echo "Watch the build logs and wait for 'Running' status."
