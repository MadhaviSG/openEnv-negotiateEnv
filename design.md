# Design — NegotiateEnv

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Training Infrastructure                 │
│  train_negotiate.py (TRL+vLLM, H100)                    │
│  train_negotiate_unsloth.py (Unsloth 4-bit, T4)         │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP / WebSocket
┌────────────────────▼────────────────────────────────────┐
│              FastAPI Server (port 7860)                  │
│  app.py → create_fastapi_app(NegotiateEnvironment)       │
│  Endpoints: /reset  /step  /state  /health               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            NegotiateEnvironment (environment.py)         │
│                                                          │
│  reset() ──► sample scenario ──► init hidden state       │
│  step()  ──► action counter ──► drift check              │
│           ──► opponent.respond() ──► compute_reward()    │
│  state() ──► step_count, difficulty, action_counter      │
└──────┬──────────────────────────┬───────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│ AEOpponent  │          │ dataset_loader  │
│ (opponent.py)│          │                 │
│ hardball    │          │ HF xlsx (200)   │
│ concession  │          │ → built-in (20) │
│ urgency     │          └─────────────────┘
│ cooperative │
└─────────────┘
```

## Key Design Decisions

### 1. Hidden Information via Server-Side State
`vendor_floor_price` is stored only in `NegotiateEnvironment._vendor_floor_price`. It is never serialized into `NegotiateObservation`, never returned by any endpoint, and never logged. The agent must infer it from the AE's counter-offers — this is the core RL challenge.

### 2. Concession Model (seat_count + contract_length as multipliers)
The AE's willingness to concede is not fixed — it scales with the value of the deal:
```python
seat_score     = min(log(seat_count) / 10, 1)      # more seats = more room
contract_score = min(contract_length / 5, 1)        # longer = more leverage
competitor_score = max(0, gap / list_price)          # competitor pressure
concession_score = 0.4*seat_score + 0.3*contract_score + 0.3*competitor_score
```
This means the agent can learn to use seat count and contract length as negotiation levers.

### 3. Reward: Terminal + Shaping + Penalties
```
total_reward = terminal_reward + shaping_reward + penalties
```
- Terminal reward is the largest signal (deal quality at close)
- Shaping rewards are small (+0.03 to +0.05) to guide mid-episode behavior
- Penalties prevent degenerate policies (looping, lowballing, early acceptance)

### 4. Opponent Strategies
Each scenario is assigned one of four strategies at load time. The agent doesn't know which strategy it's facing — it must infer from the AE's responses. This forces generalization.

### 5. Constraint Drift
A single constraint injects at `drift_turn` (scenario-specific). This breaks the assumption that the negotiation context is static, forcing the agent to adapt rather than follow a memorized script.

### 6. Difficulty via Floor Multiplier + Concession Scale
Rather than separate scenario sets per difficulty, we apply multipliers to the same scenarios:
- `floor_multiplier > 1.0` (hard) → tighter margin, harder to get a good deal
- `concession_scale < 1.0` (hard) → AE gives less per turn

### 7. Two Training Paths
- **TRL + vLLM** (`train_negotiate.py`): Uses `rollout_func` to run full multi-turn episodes, accumulates `prompt_ids + completion_ids + logprobs` across turns, passes `env_reward` as a kwarg to the reward function. Requires H100.
- **Unsloth** (`train_negotiate_unsloth.py`): Uses `reward_negotiate()` which runs a full episode inside the reward function. Turn 0 uses GRPO's pre-generated completion (so its gradient matters); turns 1+ use `model.generate` with `torch.no_grad()`. Runs on free T4.

---

## Data Flow: Training Episode (TRL path)

```
1. GRPOTrainer calls rollout_func(prompts)
2. rollout_func calls env.reset() → gets NegotiateObservation
3. format_obs_as_prompt(obs) → LLM prompt string
4. generate_rollout_completions(trainer, [prompt]) → token ids + logprobs
5. parse_llm_to_action(completion_text) → NegotiateAction
6. env.step(action) → new NegotiateObservation
7. Repeat steps 3-6 until obs.done
8. Return {prompt_ids, completion_ids, logprobs, env_reward}
9. GRPOTrainer calls reward_from_env(completions, env_reward=...)
10. GRPO loss computed, policy updated
```

## Data Flow: Training Episode (Unsloth path)

```
1. build_dataset() calls env_reset() for each of N episodes
   → formats as chat prompt → stores in Dataset
2. GRPOTrainer generates completion for turn 0 (gradient flows here)
3. reward_negotiate(completions) is called:
   a. env_reset() → fresh episode
   b. parse_to_action(completion) → env_step() [turn 0, uses GRPO completion]
   c. For turns 1+: model.generate() → parse → env_step() [no_grad]
   d. Return final obs.reward
4. GRPO loss computed, policy updated
```

## File Structure

```
negotiate_env/
├── __init__.py                  NegotiateEnv alias, exports
├── models.py                    NegotiateAction, NegotiateObservation (Pydantic)
├── scenarios.py                 20 built-in scenarios
├── dataset_loader.py            HF xlsx loader + fallback
├── client/
│   └── negotiate_env_client.py  WebSocket client, prompt formatter, action parser
└── server/
    ├── environment.py           Core RL env (reset, step, reward, action counter)
    ├── opponent.py              AEOpponent (4 strategies + concession model)
    ├── difficulty.py            DifficultyConfig dataclass
    ├── app.py                   FastAPI app via create_fastapi_app
    └── Dockerfile               Multi-stage uv build

train_negotiate.py               TRL GRPO (H100, vLLM colocate/server)
train_negotiate_unsloth.py       Unsloth 4-bit LoRA (T4)
run_agent.py                     Live LLM agent via OpenAI API
baseline_random.py               Random policy baseline
baseline_rule.py                 Rule-based policy baseline
evaluate.py                      Metrics: reward, success, strategy distribution
demo.py                          Interactive transcript demo
```

## Observation Schema

```python
class NegotiateObservation(Observation):
    context: str                    # scenario description (no floor price)
    your_max_price: float           # agent's budget ceiling
    your_max_length: float          # agent's max contract length
    your_max_cap: float             # agent's max annual increase cap
    ae_message: str                 # AE's last message
    conversation_history: list[str] # full transcript so far
    turn_number: int                # current turn
    max_turns: int                  # episode turn limit
    active_constraints: list[str]   # drift events injected so far
    current_offer: dict             # AE's current standing offer
    reward: float                   # 0.0 until episode ends
    done: bool
```

## Action Schema

```python
class NegotiateAction(Action):
    action_type: str        # counter | offer | probe | accept | walkaway
    price_per_seat: float   # proposed price (0 if probe/accept/walkaway)
    contract_length: float  # proposed years
    annual_increase_cap: float
    message: str            # natural language message to AE
```
