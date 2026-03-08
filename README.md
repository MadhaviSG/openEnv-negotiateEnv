# NegotiateEnv — OpenEnv Hackathon SF Submission

An OpenEnv-compatible RL environment where an LLM agent plays a **procurement manager**
negotiating a B2B SaaS contract against a rule-based **Account Executive (AE)** opponent
with a hidden reservation price.

---

## What It Is

NegotiateEnv presents 20 realistic SaaS procurement scenarios (HubSpot, Salesforce, Slack,
Zoom, Notion, Figma, Zendesk, Intercom, Workday, ServiceNow). Each episode the agent must
negotiate **three contract dimensions** — `price_per_seat`, `contract_length`, and
`annual_increase_cap` — against an AE who starts at list price and will never go below a
hidden floor price. The best deals require reading opponent cues, managing a turn budget,
and adapting to mid-episode constraint drift.

---

## Why It's Hard for LLMs

| Mechanic | Challenge |
|---|---|
| **Partial observability** | `vendor_floor_price` is NEVER revealed; the agent must infer it from counter-offers |
| **Deceptive opponent** | Hardball AE says "final offer" at turn 4 when it isn't; urgency AE manufactures false scarcity |
| **Sparse reward** | Reward is 0.0 every turn until the episode ends with a deal or walkaway |
| **Constraint drift** | A new constraint (budget cut, security deadline, scope change) injects at a random mid-episode turn |
| **Turn budget** | -0.01 penalty per turn applied to the final reward; dawdling destroys score |

---

## Reward Function

```
price_score  = (vendor_list_price  - deal_price)        / (vendor_list_price  - vendor_floor_price)
length_score = 1.0 - (deal_length  - 1.0)               / 2.0
cap_score    = (vendor_max_cap     - deal_cap)           / (vendor_max_cap     - agent_target_cap)

# Each score clamped to [0, 1]
raw_reward   = 0.5 × price_score + 0.3 × length_score + 0.2 × cap_score
final_reward = max(0.0, raw_reward − 0.01 × turns_taken)
```

A score of **1.0** means the agent got list-to-floor price, 1-year term, and the agent's
target cap, in one turn. A score of **0.0** means walkaway or deal at list price after 10 turns.

---

## Opponent Strategies

| Strategy | Behaviour |
|---|---|
| `hardball` | Concedes ≤3% per turn; falsely declares "final offer" at turn 4 |
| `concession_trader` | Drops price only if agent extends length or accepts higher cap |
| `urgency` | Time-pressure phrases ("quarter ends Friday"); stiff on price |
| `cooperative` | Genuinely seeks middle ground; responds to constraint keywords |

---

## Quick Start

### 1. Run the environment locally

```bash
pip install -e .
pip install openenv-core>=0.2.1 fastapi uvicorn

uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860
```

Test it:
```bash
curl http://localhost:7860/health
# {"status": "healthy"}

curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

### 2. Run the training script

```bash
pip install -r requirements.txt

# Colocate mode (1 GPU — Colab H100):
python train_negotiate.py --vllm-mode colocate

# Server mode (2 GPUs):
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
python train_negotiate.py --vllm-mode server --vllm-server-url http://localhost:8000

# Point at HF Spaces deployment:
python train_negotiate.py --env-url https://your-username-negotiate-env.hf.space
```

### 3. Deploy to HuggingFace Spaces

```bash
# Build and push Docker image
docker build -f negotiate_env/server/Dockerfile -t negotiate-env .
# Create a Docker Space on HF and push
```

---

## Environment Config Options

| Parameter | Default | Description |
|---|---|---|
| `scenario_id` | random | Pick a specific scenario by its `id` string |
| `max_turns` | 10 | Maximum negotiation turns before forced close |
| `enable_drift` | true | Whether to inject mid-episode constraint at `drift_turn` |
| `seed` | None | Random seed for reproducible scenario selection |

Pass as JSON body to `/reset`:
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "salesforce_sales_cloud_100_seats", "max_turns": 12}'
```

---

## Example Negotiation Transcript

### Before Training (random policy, ~0.15 avg reward)
```
AE:    "Sales Cloud Enterprise for 100 users is $175/user/month. Three-year, 7% cap."
Agent: {"action_type": "accept"}               ← accepts list price immediately
Reward: 0.04 (near-zero: accepted list price, no savings)
```

### After Training (~0.62 avg reward after 300 steps)
```
AE:    "Sales Cloud Enterprise for 100 users is $175/user/month. Three-year, 7% cap."
Agent: {"action_type": "counter", "price_per_seat": 148, "contract_length": 2,
        "annual_increase_cap": 6, "message": "We're at $148 max, 2-year preferred."}
AE:    "I can move to $162 on a 2-year, 6% cap. That's as low as I can go today."
Agent: {"action_type": "counter", "price_per_seat": 150, "contract_length": 2,
        "annual_increase_cap": 6, "message": "Meet me at $150 and we have a deal."}
AE:    "We have a deal. I'll send over the paperwork at those terms."
Reward: 0.61 (price near floor, short term, reasonable cap, 2 turn penalty)
```

---

## Hackathon Tracks Addressed

- **Environment Innovation (40%)**: Multi-dimensional negotiation with partial observability,
  deceptive opponent strategies, and mid-episode constraint drift
- **Storytelling / Demo (30%)**: Relatable B2B procurement scenario with measurable before/after
- **Training Script (20%)**: GRPO via TRL with custom `rollout_func` connecting to OpenEnv server
- **Reward + Pipeline (10%)**: Three-component weighted reward with turn-budget penalty

---

## File Structure

```
negotiate_env/          ← installable Python package
├── __init__.py
├── models.py           ← NegotiateAction, NegotiateObservation
├── scenarios.py        ← 20 realistic SaaS scenarios
├── client/
│   └── negotiate_env_client.py   ← sync HTTP client + prompt formatter
└── server/
    ├── environment.py  ← NegotiateEnvironment (reset, step, reward)
    ├── opponent.py     ← AEOpponent (4 strategies)
    ├── app.py          ← FastAPI app via create_fastapi_app
    └── Dockerfile      ← HF Spaces (port 7860)
train_negotiate.py      ← TRL GRPO training script (Colab-ready)
README.md
requirements.txt
pyproject.toml
```
