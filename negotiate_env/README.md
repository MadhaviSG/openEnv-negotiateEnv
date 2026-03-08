# negotiate_env

OpenEnv-compatible RL environment: an LLM agent (procurement manager) negotiates a B2B SaaS contract against a rule-based AE with hidden reservation prices.

## Install

```bash
pip install openenv-core fastapi uvicorn pydantic
# From repo root, install the package so negotiate_env is importable:
pip install -e .
# Or set PYTHONPATH to the repo root when running.
```

## Run server

```bash
# From repo root
PYTHONPATH=. uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860
```

## Use client

```python
import asyncio
from negotiate_env.client import NegotiateEnvClient
from negotiate_env.client.negotiate_env_client import observation_to_prompt, parse_llm_response_to_action
from negotiate_env.models import NegotiateAction

async def main():
    async with NegotiateEnvClient(base_url="http://localhost:7860") as client:
        result = await client.reset(seed=42)
        obs = result.observation
        while not result.done:
            prompt = observation_to_prompt(obs)
            # llm_text = your_llm.generate(prompt)
            llm_text = "I'd like to counter at $80 per seat for 2 years with a 5% cap."
            action = parse_llm_response_to_action(llm_text)
            result = await client.step(action)
            obs = result.observation
        print("Reward:", result.reward)

asyncio.run(main())
```

## Docker

From repo root:

```bash
docker build -f negotiate_env/server/Dockerfile .
docker run -p 7860:7860 <image_id>
```

## Invariants

- `vendor_floor_price` is never included in any observation.
- The AE never accepts below `vendor_floor_price`.
- Reward is non-zero only when `done=True` (except the per-turn -0.01 penalty).
- `conversation_history` grows each turn and is always returned in the observation.
- Drift event is injected exactly once at `drift_turn`.
