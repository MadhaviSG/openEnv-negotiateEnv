# Requirements — NegotiateEnv

## Functional Requirements

### FR-1: OpenEnv Interface
- Environment must expose `reset()`, `step()`, `state()` APIs
- Must be compatible with OpenEnv core (`openenv-core>=0.2.1`)
- Must serve over HTTP/WebSocket via FastAPI on port 7860

### FR-2: Dataset Integration
- Load scenarios from `mayukareddy/SyntheticSaasDataset` (HuggingFace)
- File: `saas_buyer_synthetic_200.xlsx` (200 rows)
- Columns: `id, company_size, seat_count, saas_product, vendor, list_price, competitor_price, Budget, vendor_floor_price_hidden, contract_length_months, urgency`
- `vendor_floor_price_hidden` must NEVER be exposed in any observation
- Fallback to 20 built-in scenarios if HF unavailable

### FR-3: Observation Space
Agent receives:
- `vendor`, `list_price`, `competitor_price`, `seat_count`
- `contract_length`, `budget`, `turn_number`, `negotiation_history`
- `your_max_price`, `your_max_length`, `your_max_cap`
- `active_constraints` (drift events)

Agent must NOT receive: `vendor_floor_price`

### FR-4: Action Space
| Action | Description |
|---|---|
| `counter` | Propose price / length / cap |
| `offer` | Opening offer |
| `probe` | Gather info without committing |
| `accept` | Accept current standing offer |
| `walkaway` | End negotiation (reward = 0) |

### FR-5: Vendor Concession Model
Discount flexibility computed from three signals:
```
seat_score     = min(log(seat_count) / 10, 1)
contract_score = min(contract_length / 5, 1)
competitor_score = max(0, (current_price - competitor_price) / list_price)
concession_score = 0.4 * seat_score + 0.3 * contract_score + 0.3 * competitor_score
price_drop = concession_score * random(0.1, 0.3) * (current_price - floor_price)
new_price = max(floor_price, current_price - price_drop)
```

### FR-6: Opponent Strategies
Four rule-based AE strategies:
- `hardball` — slow concessions, bluffs "final offer"
- `concession_trader` — trades price for length/cap
- `urgency` — time pressure, stiff on price
- `cooperative` — seeks middle ground

### FR-7: Reward System
```
total_reward = terminal_reward + shaping_reward + penalties

terminal_reward:
  price_score  = (list_price - deal_price) / (list_price - floor_price)
  length_score = 1.0 - (deal_length - 1.0) / 2.0
  cap_score    = (vendor_max_cap - deal_cap) / (vendor_max_cap - agent_target_cap)
  raw = 0.5 * price_score + 0.3 * length_score + 0.2 * cap_score

shaping_reward:
  vendor_price_dropped  → +0.05
  competitor_referenced → +0.03
  contract_extended     → +0.03

penalties:
  turn_penalty   = -0.01 per turn
  repeat_action  = -0.03
  lowball_offer  = -0.08 (if price < competitor * 0.75)

walk_away:
  if total_contract_value > budget * 1.1 → +0.1 (smart walkaway)
  else → -0.2 (walked from achievable deal)
```

### FR-8: Constraint Drift
- One constraint injects at scenario-specific `drift_turn`
- Examples: budget cut, deadline compression, scope change
- Agent must adapt strategy mid-episode

### FR-9: Difficulty Levels
| Level | Max Turns | Drift | Floor Multiplier | AE Concession |
|---|---|---|---|---|
| `easy` | 12 | Off | ×0.90 | ×1.5 |
| `medium` | 10 | On | ×1.00 | ×1.0 |
| `hard` | 7 | On | ×1.10 | ×0.6 |

### FR-10: RL Training Pipeline
- Train `Qwen/Qwen2.5-1.5B-Instruct` using GRPO algorithm
- Two training paths:
  - TRL + vLLM (H100, Northflank)
  - Unsloth 4-bit LoRA (free Colab T4)
- Training loop: env.reset → LLM generates action → env.step → reward → TRL updates policy

### FR-11: Baseline Agents
- Random agent (lower bound)
- Rule-based agent: probe → counter at budget → extend contract → accept/walkaway

### FR-12: Evaluation
- Run N episodes, report: mean reward, success rate, avg deal price, avg turns, strategy distribution

### FR-13: Demo
- Interactive transcript showing full negotiation turn-by-turn
- Display constraint drift events, final reward

---

## Non-Functional Requirements

### NFR-1: Deployment
- Docker image deployable to HuggingFace Spaces (port 7860)
- Deployable to Northflank H100 GPU node
- Multi-stage Dockerfile using `uv` for fast builds

### NFR-2: Compatibility
- Python >= 3.10
- OpenEnv core >= 0.2.1
- TRL >= 0.29.0
- Transformers >= 4.50.0

### NFR-3: Concurrent Sessions
- `SUPPORTS_CONCURRENT_SESSIONS = True`
- Multiple parallel training rollouts must work simultaneously

### NFR-4: Hidden State Security
- `vendor_floor_price` must never appear in any HTTP response, observation dict, or log output

### NFR-5: Graceful Fallback
- If HuggingFace dataset unavailable → fall back to built-in scenarios without crashing
