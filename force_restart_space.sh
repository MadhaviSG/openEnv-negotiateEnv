#!/bin/bash
# Force HuggingFace Space to restart and pick up new code

echo "Forcing Space restart by making a small change..."

cd hf_space

# Add a comment to README to trigger rebuild
echo "" >> README.md
echo "<!-- Updated: $(date) -->" >> README.md

git add README.md
git commit -m "Force rebuild: Update timestamp"
git push

cd ..

echo ""
echo "Space will restart in 1-2 minutes."
echo "Check status at: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env"
echo ""
echo "Wait 2 minutes, then run: python3 test_all_endpoints.py"
