#!/bin/bash
# Push your code to GitHub

echo "=================================================="
echo "Push NegotiateEnv to GitHub"
echo "=================================================="
echo ""

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: NegotiateEnv for OpenEnv Hackathon"
fi

# Add GitHub remote (replace with your actual repo URL)
echo "Setting up GitHub remote..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/kushal511/negotiate-env.git

# Push to GitHub
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main --force

echo ""
echo "=================================================="
echo "✅ Code pushed to GitHub!"
echo "=================================================="
echo ""
echo "Your repository: https://github.com/kushal511/negotiate-env"
echo ""
echo "Now in Colab, run:"
echo "  !git clone https://github.com/kushal511/negotiate-env.git"
echo "  %cd negotiate-env"
