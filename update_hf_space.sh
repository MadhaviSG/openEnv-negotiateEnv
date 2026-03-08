#!/bin/bash
# Update HuggingFace Space with fixed app.py

echo "Updating HuggingFace Space with session management fix..."

# Navigate to hf_space directory
cd hf_space

# Copy the fixed app.py
cp ../negotiate_env/server/app.py negotiate_env/server/app.py

# Commit and push
git add negotiate_env/server/app.py
git commit -m "Fix: Add session management for concurrent training"
git push

echo ""
echo "✓ Space updated!"
echo "✓ Wait 1-2 minutes for rebuild"
echo "✓ Then test: python test_all_endpoints.py"
