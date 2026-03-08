# OpenEnv Hackathon Submission Guide

**Team:** Mayuka Reddy & Kushal Adhyaru  
**Project:** NegotiateEnv — B2B SaaS Contract Negotiation RL Environment  
**Deadline:** March 8th, 2026, 1:00 PM

---

## Pre-Submission Checklist

### 1. HuggingFace Dataset ✅
- [x] Dataset uploaded: `mayukareddy/SyntheticSaasDataset`
- [x] File: `saas_buyer_synthetic_200.xlsx` (200 scenarios)
- [x] Dataset is public
- [x] Columns verified: `vendor_floor_price_hidden`, `Budget`, `contract_length_months`

### 2. HuggingFace Space (Environment Server)
- [ ] Create new Space: `mayukareddy/negotiate-env` or `openenv/negotiate-env`
- [ ] Space type: Docker
- [ ] Port: 7860
- [ ] Push Dockerfile and code
- [ ] Verify `/health` endpoint responds
- [ ] Test `/reset` and `/step` endpoints

### 3. Trained Model
- [ ] Train model using `train_negotiate.py` or `train_negotiate_unsloth.py`
- [ ] Push to HuggingFace Hub: `mayukareddy/negotiate-qwen-1.5b-grpo`
- [ ] Include model card with training details
- [ ] Record final metrics (avg reward, success rate)

### 4. GitHub Repository
- [ ] Repo is public: `github.com/mayuka-reddy/negotiate-env` or similar
- [ ] README.md complete with all sections
- [ ] All code files committed
- [ ] requirements.txt and pyproject.toml up to date
- [ ] Documentation files included (requirements.md, design.md, tasks.md)

### 5. Demo & Evaluation
- [ ] Run `demo.py` and capture transcript
- [ ] Run `evaluate.py` for all baselines (random, rule, llm)
- [ ] Generate plots: `plot_reward_curve.py`, `plot_strategy_distribution.py`
- [ ] Save results in `results/` folder

### 6. Northflank GPU Access
- [ ] Submit GPU request form: https://docs.google.com/forms/d/e/1FAIpQLSd2bxx5jAXE8D3FjF7OVekSxwpDVMf1LWE3Z-g4FZoDJ4W6xg/viewform
- [ ] Receive Northflank project workspace
- [ ] Test deployment on H100

---

## Deployment Instructions

### Deploy to HuggingFace Spaces

#### Option 1: Web UI
1. Go to https://huggingface.co/new-space
2. Space name: `negotiate-env`
3. License: MIT
4. Space SDK: Docker
5. Upload files:
   - `negotiate_env/` (entire folder)
   - `pyproject.toml`
   - `negotiate_env/server/Dockerfile` → rename to `Dockerfile` at root
6. Space will auto-build and expose port 7860

#### Option 2: Git Push
```bash
# Clone your space
git clone https://huggingface.co/spaces/mayukareddy/negotiate-env
cd negotiate-env

# Copy project files
cp -r ../negotiate_env .
cp ../pyproject.toml .
cp ../negotiate_env/server/Dockerfile ./Dockerfile

# Create README.md for the Space
cat > README.md << 'EOF'
---
title: NegotiateEnv
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# NegotiateEnv — OpenEnv B2B SaaS Negotiation

OpenEnv-compatible RL environment for training LLM agents to negotiate enterprise SaaS contracts.

## Endpoints

- `GET /health` — Health check
- `POST /reset` — Start new negotiation episode
- `POST /step` — Execute agent action
- `GET /state` — Get environment state

## Usage

```python
from negotiate_env import NegotiateEnv

env = NegotiateEnv(base_url="https://mayukareddy-negotiate-env.hf.space")
obs = env.reset()
obs = env.step(action)
```

See full documentation: https://github.com/mayuka-reddy/negotiate-env
EOF

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

#### Verify Deployment
```bash
# Check health
curl https://mayukareddy-negotiate-env.hf.space/health

# Test reset
curl -X POST https://mayukareddy-negotiate-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Should return NegotiateObservation JSON
```

### Deploy to Northflank (H100 Training)

1. **Create Project** (via Northflank dashboard after GPU request approved)

2. **Deploy via GitHub**
   - Connect GitHub repo
   - Select branch: `main`
   - Build type: Dockerfile
   - Dockerfile path: `negotiate_env/server/Dockerfile`
   - Port: 7860

3. **Deploy Training Job**
   ```bash
   # SSH into Northflank container
   northflank ssh --project negotiate-env --service training
   
   # Inside container
   git clone https://github.com/mayuka-reddy/negotiate-env.git
   cd negotiate-env
   pip install -e .
   pip install -r requirements.txt
   
   # Start environment server in background
   uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860 &
   
   # Run training
   accelerate launch --config_file accelerate_config.yaml train_negotiate.py \
     --vllm-mode colocate \
     --num-episodes 300 \
     --output-dir negotiate-grpo-output
   ```

4. **Monitor Training**
   ```bash
   # Watch logs
   tail -f negotiate-grpo-output/training.log
   
   # Plot reward curve
   python plot_reward_curve.py \
     --log-file negotiate-grpo-output/trainer_state.json \
     --out reward_curve.png
   ```

---

## Training Instructions

### Quick Training (Unsloth on Free Colab T4)

1. Open Google Colab: https://colab.research.google.com/
2. Runtime → Change runtime type → T4 GPU
3. Run:
```python
!git clone https://github.com/mayuka-reddy/negotiate-env.git
%cd negotiate-env
!pip install -e .
!pip install "unsloth[colab-new]" trl datasets requests

# Start environment server
!uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860 &

# Wait for server to start
import time
time.sleep(10)

# Train
!python train_negotiate_unsloth.py \
  --env-url http://localhost:7860 \
  --num-episodes 100 \
  --output-dir negotiate-unsloth-output
```

### Full Training (TRL GRPO on Northflank H100)

```bash
# On Northflank H100 node
accelerate launch --config_file accelerate_config.yaml train_negotiate.py \
  --vllm-mode colocate \
  --num-episodes 300 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --output-dir negotiate-grpo-h100
```

Expected training time: ~2-3 hours on H100

---

## Evaluation & Demo

### Run Baselines
```bash
# Start environment server
uvicorn negotiate_env.server.app:app --port 7860 &

# Random baseline
python baseline_random.py --episodes 100 > results/baseline_random.txt

# Rule-based baseline
python baseline_rule.py --episodes 100 > results/baseline_rule.txt

# Evaluate with metrics
python evaluate.py --agent random --episodes 100
python evaluate.py --agent rule --episodes 100
```

### Run Demo
```bash
# Rule-based agent demo
python demo.py --difficulty medium

# LLM agent demo (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python demo.py --agent llm --model gpt-4o-mini --difficulty hard
```

### Generate Plots
```bash
# Reward curve (after training)
python plot_reward_curve.py \
  --log-file negotiate-grpo-output/trainer_state.json \
  --out results/reward_curve.png

# Strategy distribution comparison
python plot_strategy_distribution.py \
  --before-agent rule \
  --after-agent llm \
  --after-model gpt-4o-mini \
  --episodes 100 \
  --out results/strategy_distribution.png
```

---

## Submission Package

Create a `submission/` folder with:

```
submission/
├── README.md                          # Project overview
├── demo_transcript.txt                # Output from demo.py
├── evaluation_results.txt             # Output from evaluate.py
├── reward_curve.png                   # Training progress plot
├── strategy_distribution.png          # Before/after comparison
├── model_card.md                      # Trained model details
└── links.txt                          # HF Space, Dataset, Model URLs
```

### Create Submission Package
```bash
mkdir -p submission

# Copy README
cp README.md submission/

# Run demo and save output
python demo.py --difficulty medium > submission/demo_transcript.txt

# Run evaluation
python evaluate.py --agent rule --episodes 100 > submission/evaluation_results.txt

# Generate plots (if training completed)
python plot_reward_curve.py \
  --log-file negotiate-grpo-output/trainer_state.json \
  --out submission/reward_curve.png

python plot_strategy_distribution.py \
  --before-agent rule \
  --after-agent llm \
  --episodes 50 \
  --out submission/strategy_distribution.png

# Create links file
cat > submission/links.txt << EOF
HuggingFace Space: https://huggingface.co/spaces/mayukareddy/negotiate-env
HuggingFace Dataset: https://huggingface.co/datasets/mayukareddy/SyntheticSaasDataset
Trained Model: https://huggingface.co/mayukareddy/negotiate-qwen-1.5b-grpo
GitHub Repository: https://github.com/mayuka-reddy/negotiate-env
EOF

# Create model card
cat > submission/model_card.md << EOF
# negotiate-qwen-1.5b-grpo

## Model Details
- Base Model: Qwen/Qwen2.5-1.5B-Instruct
- Training Algorithm: GRPO (Group Relative Policy Optimization)
- Environment: NegotiateEnv (OpenEnv-compatible)
- Training Episodes: 300
- Hardware: NVIDIA H100 (Northflank)

## Performance
- Average Reward: 0.62 (baseline: 0.15)
- Success Rate: 78% (baseline: 52%)
- Average Deal Price: \$142/seat (vs \$165 baseline)
- Average Turns: 3.8 (vs 1.2 baseline)

## Strategy Learned
- Probes first (22% vs 6% baseline)
- Counters with numbers (48% vs 12%)
- Accepts less frequently (14% vs 52%)
- Smart walkaways (16% vs 30%)

## Usage
\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mayukareddy/negotiate-qwen-1.5b-grpo")
tokenizer = AutoTokenizer.from_pretrained("mayukareddy/negotiate-qwen-1.5b-grpo")

# Use with NegotiateEnv
from negotiate_env import NegotiateEnv
env = NegotiateEnv()
# ... run episodes
\`\`\`
EOF

echo "Submission package created in submission/"
```

---

## Final Checks Before Submission

### Environment Server
- [ ] HF Space is live
- [ ] `/health` returns `{"status": "healthy"}`
- [ ] `/reset` returns valid NegotiateObservation
- [ ] `/step` accepts NegotiateAction and returns observation
- [ ] `vendor_floor_price` is NOT in any response

### Dataset
- [ ] Dataset is public on HuggingFace
- [ ] 200 rows load correctly
- [ ] Column names match: `vendor_floor_price_hidden`, `Budget`, `contract_length_months`

### Training
- [ ] Model trained for at least 100 episodes
- [ ] Reward curve shows improvement
- [ ] Model pushed to HuggingFace Hub
- [ ] Model card includes metrics

### Demo & Evaluation
- [ ] `demo.py` runs without errors
- [ ] Transcript shows multi-turn negotiation
- [ ] `evaluate.py` shows metrics for all baselines
- [ ] Plots generated and saved

### Documentation
- [ ] README.md is comprehensive
- [ ] All code is commented
- [ ] requirements.txt is complete
- [ ] GitHub repo is public

### Submission
- [ ] All files in `submission/` folder
- [ ] Links file has all URLs
- [ ] Demo transcript is readable
- [ ] Evaluation results show improvement

---

## Hackathon Scoring Alignment

### Environment Innovation (40%)
- ✅ Partial observability (hidden vendor floor price)
- ✅ Deceptive opponent (4 strategies, false "final offer")
- ✅ Constraint drift (mid-episode changes)
- ✅ 3-dimensional action space (price, length, cap)
- ✅ Multi-turn reasoning required

### Storytelling / Demo (30%)
- ✅ Relatable B2B procurement scenario
- ✅ Before/after training comparison
- ✅ Interactive demo script
- ✅ Clear reward improvement narrative

### Training Script (20%)
- ✅ TRL GRPO with custom rollout function
- ✅ Unsloth fallback for T4
- ✅ Multi-turn episode handling
- ✅ Reward curve logging

### Reward + Pipeline (10%)
- ✅ Terminal + shaping + penalties
- ✅ Budget awareness
- ✅ Walk-away logic
- ✅ Strategy discovery metrics

---

## Support & Resources

- **OpenEnv Docs**: https://meta-pytorch.org/OpenEnv/
- **TRL Docs**: https://huggingface.co/docs/trl/en/openenv
- **Northflank Guide**: https://northflank.notion.site/Deploy-AI-projects-with-Northflank-1a76d14c7851805f8a0ecc780fa33547
- **Hackathon Discord**: Ask questions in #openenv-hackathon channel

---

## Contact

- Mayuka Reddy: mayukareddy10@gmail.com
- Kushal Adhyaru: kushaladhyaru5112001@gmail.com

Good luck! 🚀
