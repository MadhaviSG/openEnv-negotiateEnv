# Tasks — NegotiateEnv

Status: ✅ Done | 🔧 In Progress | ❌ Not Started

---

## Environment Core

| # | Task | Status | File |
|---|---|---|---|
| 1 | OpenEnv `reset()` / `step()` / `state()` interface | ✅ | `negotiate_env/server/environment.py` |
| 2 | `NegotiateAction` and `NegotiateObservation` Pydantic models | ✅ | `negotiate_env/models.py` |
| 3 | Hidden `vendor_floor_price` — never in observation | ✅ | `environment.py` |
| 4 | 20 built-in B2B SaaS scenarios | ✅ | `negotiate_env/scenarios.py` |
| 5 | HuggingFace dataset loader (`mayukareddy/SyntheticSaasDataset`) | ✅ | `negotiate_env/dataset_loader.py` |
| 6 | xlsx parsing with correct column names (`vendor_floor_price_hidden`, `Budget`, `contract_length_months`) | ✅ | `negotiate_env/dataset_loader.py` |
| 7 | Constraint drift injection at `drift_turn` | ✅ | `environment.py` |
| 8 | Action counter for strategy discovery metrics | ✅ | `environment.py` |
| 9 | Difficulty levels: easy / medium / hard | ✅ | `negotiate_env/server/difficulty.py` |
| 10 | `SUPPORTS_CONCURRENT_SESSIONS = True` | ✅ | `environment.py` |

---

## Vendor Concession Model

| # | Task | Status | File |
|---|---|---|---|
| 11 | Seat count effect (`seat_score = min(log(seat_count)/10, 1)`) | ✅ | `opponent.py` |
| 12 | Contract length effect (`contract_score = min(length/5, 1)`) | ✅ | `opponent.py` |
| 13 | Competitor pressure effect | ✅ | `opponent.py` |
| 14 | Combined `concession_score` → price movement | ✅ | `opponent.py` |
| 15 | `hardball` strategy | ✅ | `opponent.py` |
| 16 | `concession_trader` strategy | ✅ | `opponent.py` |
| 17 | `urgency` strategy | ✅ | `opponent.py` |
| 18 | `cooperative` strategy | ✅ | `opponent.py` |

---

## Reward System

| # | Task | Status | File |
|---|---|---|---|
| 19 | Terminal reward (price + length + cap weighted score) | ✅ | `environment.py` → `_compute_reward()` |
| 20 | Turn penalty (`-0.01` per turn) | ✅ | `environment.py` |
| 21 | Shaping reward (vendor price drop, competitor reference, contract extension) | ✅ | `environment.py` — `+0.05` price drop, `+0.03` competitor ref, `+0.03` contract extension |
| 22 | Repeat action penalty (`-0.03`) | ✅ | `environment.py` — tracked via `_last_action_type` |
| 23 | Lowball penalty (`-0.08` if price < competitor × 0.75) | ✅ | `environment.py` — uses `scenario.competitor_price` |
| 24 | Budget awareness (total contract value vs budget) | ✅ | `environment.py` — `+0.3` within budget, `-0.4` over budget |
| 25 | Walk-away reward logic (`+0.1` smart / `-0.2` bad) | ✅ | `environment.py` — smart if standing offer > 110% of budget |

---

## Server & API

| # | Task | Status | File |
|---|---|---|---|
| 26 | FastAPI server via `create_fastapi_app` | ✅ | `negotiate_env/server/app.py` |
| 27 | `/reset`, `/step`, `/state`, `/health` endpoints | ✅ | `app.py` |
| 28 | Difficulty + HF dataset via env vars (`NEGOTIATE_DIFFICULTY`, `NEGOTIATE_USE_HF_DATASET`) | ✅ | `app.py` |
| 29 | Dockerfile (multi-stage, uv-based) | ✅ | `negotiate_env/server/Dockerfile` |
| 30 | WebSocket client with typed `reset()` / `step()` | ✅ | `negotiate_env/client/negotiate_env_client.py` |
| 31 | Prompt formatter (`observation_to_prompt`) | ✅ | `negotiate_env_client.py` |
| 32 | LLM output parser (JSON → heuristic fallback) | ✅ | `negotiate_env_client.py` |

---

## Training

| # | Task | Status | File |
|---|---|---|---|
| 33 | TRL GRPO training script (H100, vLLM colocate/server) | ✅ | `train_negotiate.py` |
| 34 | `rollout_func` — multi-turn episode accumulates token ids + logprobs | ✅ | `train_negotiate.py` |
| 35 | `reward_from_env` — extracts env reward from rollout kwargs | ✅ | `train_negotiate.py` |
| 36 | Unsloth 4-bit LoRA training script (T4) | ✅ | `train_negotiate_unsloth.py` |
| 37 | `reward_negotiate` — full episode inside reward fn, turn 0 uses GRPO completion | ✅ | `train_negotiate_unsloth.py` |
| 38 | Save merged 16-bit model after Unsloth training | ✅ | `train_negotiate_unsloth.py` |
| 39 | Training reward curve logging (episode vs reward) | ✅ | `plot_reward_curve.py` — reads `trainer_state.json`, plots MA-smoothed curve |
| 40 | Strategy distribution plot (before vs after training) | ✅ | `plot_strategy_distribution.py` — side-by-side bar chart, live or JSON input |
| 41 | `accelerate launch` config for Northflank H100 | ✅ | `accelerate_config.yaml` — bf16, single-GPU; comment shows multi-GPU option |

---

## Baselines & Evaluation

| # | Task | Status | File |
|---|---|---|---|
| 42 | Random agent baseline | ✅ | `baseline_random.py` |
| 43 | Rule-based agent baseline | ✅ | `baseline_rule.py` |
| 44 | Evaluation script (reward, success rate, avg price, turns, strategy freq) | ✅ | `evaluate.py` |
| 45 | Compare RL agent vs baselines in evaluate.py | ✅ | `evaluate.py` — `--agent llm` option |
| 46 | Demo script with full transcript + drift display | ✅ | `demo.py` |
| 47 | Live LLM agent runner (OpenAI API) | ✅ | `run_agent.py` |

---

## Documentation

| # | Task | Status | File |
|---|---|---|---|
| 48 | README with full project overview | ✅ | `README.md` |
| 49 | Requirements document | ✅ | `requirements.md` |
| 50 | Design document | ✅ | `design.md` |
| 51 | Tasks document | ✅ | `tasks.md` |

---

## Quick Status Summary

- Core environment: **100% complete**
- Vendor concession model: **100% complete**
- Reward system: **100% complete** (terminal + turn penalty + shaping + penalties + budget + walkaway)
- Server & API: **100% complete**
- Training scripts: **100% complete** (scripts + reward curve plot + strategy distribution + accelerate config)
- Baselines & evaluation: **100% complete**
- Documentation: **100% complete**
