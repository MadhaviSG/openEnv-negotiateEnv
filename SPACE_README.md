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

**Built for OpenEnv Hackathon SF 2024**

## Features

- 🎯 Hidden information game (vendor floor price never revealed)
- 🤖 4 distinct opponent strategies (hardball, concession_trader, urgency, cooperative)
- 📊 200 scenarios from HuggingFace dataset
- 🎲 Constraint drift (mid-episode changes)
- 🏆 Complex reward shaping (terminal + shaping + penalties)

## API Endpoints

- `GET /health` - Health check
- `POST /reset` - Reset environment and get initial observation
- `POST /step` - Take an action and get next observation
- `GET /state` - Get current environment state

## Example Usage

```python
import requests

# Reset environment
response = requests.post("https://kushaladhyaru-negotiate-env.hf.space/reset", json={})
obs = response.json()
print(f"Scenario: {obs['context']}")
print(f"Your budget: ${obs['your_max_price']}")

# Take action
action = {
    "action_type": "counter",
    "price_per_seat": 150.0,
    "contract_length": 2.0,
    "annual_increase_cap": 5.0,
    "message": "Can we do $150/seat for 2 years?"
}
response = requests.post("https://kushaladhyaru-negotiate-env.hf.space/step", json=action)
obs = response.json()
print(f"AE response: {obs['ae_message']}")
print(f"Done: {obs['done']}, Reward: {obs['reward']}")
```

## Training

Use this environment with TRL GRPO or Unsloth for training LLM agents.

See the [GitHub repository](https://github.com/kushal511/negotiate-env) for training scripts and Colab notebooks.

## Team

- Mayuka Reddy — [mayuka-reddy](https://github.com/mayuka-reddy)
- Kushal Adhyaru — [kushal511](https://github.com/kushal511)

## Dataset

Scenarios from [mayukareddy/SyntheticSaasDataset](https://huggingface.co/datasets/mayukareddy/SyntheticSaasDataset)
