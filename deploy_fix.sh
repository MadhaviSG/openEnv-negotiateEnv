#!/bin/bash
echo "Deploying /step endpoint fix..."
cd hf_space
git add negotiate_env/server/app_wrapper.py Dockerfile
git commit -m "Fix: Add working /step endpoint with session management"
git push
cd ..

echo ""
echo "Waiting for rebuild (90 seconds)..."
sleep 90

echo ""
echo "Testing all endpoints..."
python3 test_all_endpoints.py
