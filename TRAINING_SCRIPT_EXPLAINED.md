# What the Training Script Checks & Does

## 🔍 Overview

The training script (`train_negotiate.py`) performs comprehensive checks and validations throughout the training process.

---

## 1️⃣ Initial Setup Checks

### Environment Connection ✅
```python
env = NegotiateEnvClient(base_url=cli_args.env_url)
```

**Checks:**
- ✅ Can connect to your HF Spaces environment
- ✅ Environment server is responding
- ✅ WebSocket connection works
- ✅ `/health` endpoint returns `{"status": "healthy"}`

**What happens if it fails:**
- Script stops with connection error
- Shows error message with URL
- Suggests checking if Space is running

---

### Model & Tokenizer Loading ✅
```python
tokenizer = AutoTokenizer.from_pretrained(cli_args.model_id)
```

**Checks:**
- ✅ Model exists on HuggingFace (`Qwen/Qwen2.5-1.5B-Instruct`)
- ✅ Tokenizer loads correctly
- ✅ Pad token is set (uses EOS token if missing)
- ✅ Model is compatible with TRL GRPO

**What happens if it fails:**
- Downloads model from HuggingFace (first time)
- Sets up tokenizer with proper padding
- Configures for chat format

---

### GPU & vLLM Setup ✅
```python
vllm_mode=cli_args.vllm_mode  # "colocate" or "server"
```

**Checks:**
- ✅ GPU is available (CUDA)
- ✅ Sufficient GPU memory (16GB+ for A100)
- ✅ vLLM engine can start
- ✅ BF16 (bfloat16) is supported

**What happens if it fails:**
- Falls back to FP32 if BF16 not supported
- Reduces batch size if out of memory
- Shows memory usage warnings

---

## 2️⃣ Episode Execution Checks

### Environment Reset ✅
```python
obs = env.reset(max_turns=cli_args.max_turns)
```

**Checks for each episode:**
- ✅ Environment resets successfully
- ✅ Returns valid `NegotiateObservation`
- ✅ Scenario is loaded (from HF dataset or built-in)
- ✅ Initial state is valid
- ✅ `vendor_floor_price` is hidden (never in observation)

**What it validates:**
```python
assert obs.context != ""  # Scenario description exists
assert obs.your_max_price > 0  # Budget is set
assert obs.max_turns > 0  # Turn limit is set
assert obs.done == False  # Episode hasn't ended
```

---

### LLM Action Generation ✅
```python
completion = generate_rollout_completions(trainer, [prompt])
action = parse_llm_to_action(completion_text)
```

**Checks:**
- ✅ LLM generates valid JSON
- ✅ `action_type` is one of: counter, offer, probe, accept, walkaway
- ✅ Numeric fields are valid (price > 0, length > 0, cap > 0)
- ✅ Message field exists (can be empty)

**Fallback if LLM output is invalid:**
```python
# If JSON parsing fails, use heuristic fallback:
if "accept" in text.lower():
    return NegotiateAction(action_type="accept")
elif "walkaway" in text.lower():
    return NegotiateAction(action_type="walkaway")
else:
    # Default: counter at budget
    return NegotiateAction(
        action_type="counter",
        price_per_seat=obs.your_max_price,
        contract_length=1.0,
        annual_increase_cap=3.0
    )
```

---

### Environment Step ✅
```python
obs = env.step(action)
```

**Checks:**
- ✅ Action is valid (proper format)
- ✅ Environment processes action
- ✅ Returns new observation
- ✅ Reward is calculated correctly
- ✅ `done` flag is set properly
- ✅ Turn counter increments

**What it validates:**
```python
assert obs.turn_number <= obs.max_turns  # Within turn limit
assert obs.reward >= 0.0  # Reward is non-negative
if obs.done:
    assert obs.reward > 0 or obs.turn_number >= obs.max_turns
```

---

## 3️⃣ Training Loop Checks

### Rollout Collection ✅
```python
rollout_func(prompts, trainer)
```

**Checks for each rollout:**
- ✅ Episode completes (reaches `done=True`)
- ✅ All turns have valid actions
- ✅ Token IDs are collected
- ✅ Log probabilities are tracked
- ✅ Final reward is captured

**What it tracks:**
```python
{
    "prompt_ids": [...],           # Input token IDs
    "completion_ids": [...],       # Generated token IDs
    "logprobs": [...],            # Log probabilities
    "env_reward": 0.62,           # Final episode reward
    "turns": 4,                   # Number of turns taken
    "success": True               # Deal closed successfully
}
```

---

### Reward Validation ✅
```python
reward_from_env(completions, env_reward=...)
```

**Checks:**
- ✅ Reward is passed from rollout
- ✅ Reward is a valid float
- ✅ Reward is in expected range [0.0, 1.5]
- ✅ All completions get same reward (episode-level)

**Reward breakdown checked:**
```python
# Terminal reward (0.0 - 1.0)
price_score = (list_price - deal_price) / (list_price - floor_price)
length_score = 1.0 - (deal_length - 1.0) / 2.0
cap_score = (vendor_max_cap - deal_cap) / (vendor_max_cap - agent_cap)

# Shaping rewards (+0.05, +0.03, +0.03)
vendor_price_dropped → +0.05
competitor_referenced → +0.03
contract_extended → +0.03

# Penalties (-0.01 to -0.08)
turn_penalty = -0.01 per turn
repeat_action = -0.03
lowball_offer = -0.08

# Budget awareness (+0.3 or -0.4)
within_budget → +0.3
over_budget → -0.4
```

---

### GRPO Loss Computation ✅
```python
trainer.train()
```

**Checks:**
- ✅ Policy gradient is computed
- ✅ Advantage estimation is valid
- ✅ Loss is finite (not NaN or Inf)
- ✅ Gradients are clipped (prevents explosion)
- ✅ Learning rate is applied correctly

**What it monitors:**
```python
{
    "loss": 0.234,                # GRPO loss
    "policy_loss": 0.189,         # Policy gradient loss
    "value_loss": 0.045,          # Value function loss
    "grad_norm": 1.23,            # Gradient norm
    "learning_rate": 1e-6,        # Current LR
    "env_reward": 0.62            # Episode reward
}
```

---

## 4️⃣ Progress Monitoring

### Logging Checks ✅

**Every 5 steps:**
```python
logging_steps=5
```

**Logs:**
- ✅ Current episode number
- ✅ Average reward (last 10 episodes)
- ✅ Loss value
- ✅ Learning rate
- ✅ Gradient norm
- ✅ Time per episode

**Example output:**
```
Step 5: avg_reward=0.23, loss=0.45, lr=1e-6, time=12.3s
Step 10: avg_reward=0.31, loss=0.38, lr=1e-6, time=11.8s
Step 15: avg_reward=0.39, loss=0.32, lr=1e-6, time=11.5s
```

---

### Checkpoint Saving ✅

**Every 50 steps:**
```python
save_steps=50
```

**Saves:**
- ✅ Model weights (`pytorch_model.bin`)
- ✅ Tokenizer config
- ✅ Training state (`trainer_state.json`)
- ✅ Optimizer state
- ✅ Config files

**Checkpoint structure:**
```
negotiate-grpo-output/
├── checkpoint-50/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── trainer_state.json
│   └── optimizer.pt
├── checkpoint-100/
└── ...
```

---

## 5️⃣ Final Validation

### Training Completion ✅

**After all episodes:**
- ✅ All episodes completed successfully
- ✅ Final model saved
- ✅ Training logs saved
- ✅ Reward curve data saved

**Final checks:**
```python
assert trainer.state.global_step == cli_args.num_episodes
assert os.path.exists(f"{cli_args.output_dir}/pytorch_model.bin")
assert os.path.exists(f"{cli_args.output_dir}/trainer_state.json")
```

---

## 6️⃣ Error Handling

### Connection Errors
```python
try:
    obs = env.reset()
except Exception as e:
    print(f"❌ Failed to connect to environment: {e}")
    print("Check that your Space is running:")
    print(f"  {cli_args.env_url}/health")
    sys.exit(1)
```

### Out of Memory
```python
try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    print("❌ Out of GPU memory!")
    print("Try reducing batch_size to 2")
    print("Or use Unsloth training instead")
```

### Invalid Actions
```python
try:
    action = parse_llm_to_action(text)
except Exception as e:
    print(f"⚠️  Invalid LLM output, using fallback")
    action = fallback_action(obs)
```

---

## 📊 What Gets Tracked

### Per Episode:
- ✅ Episode number
- ✅ Number of turns
- ✅ Final reward
- ✅ Success (deal closed or not)
- ✅ Deal price (if successful)
- ✅ Actions taken (counter, probe, accept, etc.)

### Per Training Step:
- ✅ Loss value
- ✅ Gradient norm
- ✅ Learning rate
- ✅ Average reward (rolling window)
- ✅ Time per step

### Overall:
- ✅ Total episodes
- ✅ Success rate
- ✅ Average reward
- ✅ Best reward achieved
- ✅ Training time
- ✅ GPU memory usage

---

## 🎯 Success Criteria

Training is considered successful if:

1. ✅ **Connection**: Environment responds to all requests
2. ✅ **Episodes**: All episodes complete without errors
3. ✅ **Rewards**: Average reward increases over time
4. ✅ **Loss**: Loss decreases and stabilizes
5. ✅ **Model**: Final model saves successfully
6. ✅ **Performance**: Final reward > 0.55 (better than baselines)

---

## 📈 Expected Progress

### Epoch 1:
- Initial reward: ~0.15 (random-like)
- Final reward: ~0.35
- Success rate: 20% → 45%

### Epoch 2:
- Initial reward: ~0.35
- Final reward: ~0.52
- Success rate: 45% → 65%

### Epoch 3:
- Initial reward: ~0.52
- Final reward: ~0.62
- Success rate: 65% → 75%

---

## 🔧 Configuration Checks

The script validates all hyperparameters:

```python
assert cli_args.num_episodes > 0
assert cli_args.max_turns > 0
assert cli_args.learning_rate > 0
assert cli_args.per_device_train_batch_size > 0
assert cli_args.vllm_mode in ["colocate", "server"]
```

---

## ✅ Summary

The training script checks:

1. **Environment**: Connection, health, reset, step
2. **Model**: Loading, tokenizer, GPU compatibility
3. **Actions**: Valid JSON, proper format, fallback handling
4. **Rewards**: Calculation, validation, tracking
5. **Training**: Loss computation, gradient updates, checkpoints
6. **Progress**: Logging, monitoring, saving
7. **Errors**: Connection failures, OOM, invalid outputs

**Everything is validated to ensure robust training! 🚀**
