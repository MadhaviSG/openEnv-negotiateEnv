# Commands to Run in Google Colab

## Step 1: Clone Repository

```python
# Clone your GitHub repository
!git clone https://github.com/MadhaviSG/openEnv-negotiateEnv.git

# Change to the directory
%cd openEnv-negotiateEnv

# Verify files are there
!ls -la
```

## Step 2: Install Dependencies

```python
# Install the package
!pip install -e .

# Install training dependencies
!pip install requests matplotlib numpy
```

## Step 3: Test Connection to Your Environment

```python
# Set your environment URL
ENV_URL = "https://kushaladhyaru-negotiate-env.hf.space"

# Test connection
import requests
response = requests.get(f"{ENV_URL}/health")
print(response.json())  # Should print: {'status': 'healthy'}
```

## Step 4: Install TRL Dependencies (for A100)

```python
!pip install trl>=0.29.0 transformers>=4.50.0 accelerate>=0.34.0 peft>=0.13.0
!pip install vllm>=0.6.0 torch>=2.5.0
```

## Step 5: Run Training

```python
# Run TRL GRPO training
!python train_negotiate.py \
    --vllm-mode colocate \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir negotiate-grpo-output \
    --num-train-epochs 3 \
    --per-device-train-batch-size 4 \
    --learning-rate 5e-7 \
    --env-url {ENV_URL}
```

## Alternative: Use the Notebook

Instead of running commands manually, you can:

1. Upload `colab_training.ipynb` to Colab
2. Run all cells
3. Everything is automated!

---

## Quick Copy-Paste for Colab

```python
# All-in-one setup
!git clone https://github.com/MadhaviSG/openEnv-negotiateEnv.git
%cd openEnv-negotiateEnv
!pip install -e .
!pip install trl>=0.29.0 transformers>=4.50.0 accelerate>=0.34.0 peft>=0.13.0 vllm>=0.6.0 torch>=2.5.0 requests matplotlib numpy

# Set environment URL
ENV_URL = "https://kushaladhyaru-negotiate-env.hf.space"

# Test connection
import requests
print(requests.get(f"{ENV_URL}/health").json())

# Run training
!python train_negotiate.py --vllm-mode colocate --env-url {ENV_URL}
```

---

## Your Repository

**GitHub**: https://github.com/MadhaviSG/openEnv-negotiateEnv  
**HF Space**: https://kushaladhyaru-negotiate-env.hf.space

---

## Troubleshooting

### "No such file or directory: train_negotiate.py"
- Make sure you ran `%cd openEnv-negotiateEnv`
- Check files: `!ls -la`

### "Failed to connect to environment"
- Check Space is running: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env
- Test health: `!curl https://kushaladhyaru-negotiate-env.hf.space/health`

### "Out of memory"
- Make sure you selected A100 GPU
- Runtime → Change runtime type → A100
- Or reduce batch size: `--per-device-train-batch-size 2`
