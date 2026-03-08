#!/usr/bin/env python3
"""
NegotiateEnv GRPO Training Script
Trains Qwen/Qwen2.5-1.5B-Instruct to negotiate B2B SaaS contracts.

Install:
    pip install "openenv-core>=0.2.1" trl>=0.29.0 transformers>=4.50.0 vllm datasets requests

Run (colocate mode, 1 GPU — recommended for Colab H100):
    python train_negotiate.py --vllm-mode colocate

Run (server mode, 2 GPUs):
    # Terminal 1: trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
    # Terminal 2: python train_negotiate.py --vllm-mode server --vllm-server-url http://localhost:8000

Environment server:
    # Local: uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860
    # HF Spaces: set --env-url https://your-space.hf.space
"""

import argparse
import json
import re

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from negotiate_env import NegotiateEnvironment, NegotiateAction, NegotiateObservation
from negotiate_env.client.negotiate_env_client import NegotiateEnvClient

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train NegotiateEnv with TRL GRPO")
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument(
    "--env-url",
    default="http://0.0.0.0:7860",
    help="URL of deployed NegotiateEnv HF Space or local server",
)
parser.add_argument(
    "--vllm-mode",
    default="colocate",
    choices=["colocate", "server"],
    help="'colocate' shares GPU with trainer (1 GPU); 'server' uses a separate vLLM process",
)
parser.add_argument("--vllm-server-url", default="http://localhost:8000")
parser.add_argument("--max-turns", type=int, default=10)
parser.add_argument("--num-episodes", type=int, default=500)
parser.add_argument("--output-dir", default="negotiate-grpo-output")
cli_args = parser.parse_args()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert procurement manager negotiating a B2B SaaS contract.
Your goal is to get the best deal: lowest price per seat, shortest contract length,
and lowest annual price increase cap.

You will receive the current negotiation state and must respond with a JSON action:
{"action_type": "counter", "price_per_seat": 85.0, "contract_length": 1.0,
 "annual_increase_cap": 3.0, "message": "Your message to the AE"}

action_type must be one of: "offer", "counter", "probe", "accept", "walkaway"
- Use "counter" to propose specific numbers on all 3 dimensions
- Use "probe" to gather information without committing to numbers
- Use "accept" to accept the AE's current standing offer (triggers reward calculation)
- Use "walkaway" to end the negotiation (reward = 0.0)

Always respond with valid JSON. Think step by step before deciding."""

# ---------------------------------------------------------------------------
# Global objects (set up after CLI parsed, before trainer constructed)
# ---------------------------------------------------------------------------

tokenizer: AutoTokenizer = None  # type: ignore[assignment]
env: NegotiateEnvClient = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_obs_as_prompt(obs: NegotiateObservation) -> str:
    """Format observation as a clean negotiation-state string for the LLM."""
    turns_remaining = obs.max_turns - obs.turn_number
    current = obs.current_offer
    history_tail = obs.conversation_history[-4:] if obs.conversation_history else []

    lines = [
        "## Negotiation Context",
        obs.context,
        "",
        "## AE's Current Offer",
        f"  Price:    ${current.get('price_per_seat', 0):.2f}/seat/month",
        f"  Length:   {current.get('contract_length', 0):.0f} year(s)",
        f"  Ann. cap: {current.get('annual_increase_cap', 0):.1f}%",
        "",
        "## Your Budget Limits",
        f"  Max price:  ${obs.your_max_price:.2f}/seat/month",
        f"  Max length: {obs.your_max_length:.0f} year(s)",
        f"  Max cap:    {obs.your_max_cap:.1f}%",
    ]

    if obs.active_constraints:
        lines += ["", "## ⚠ Active Constraints"]
        lines += [f"  - {c}" for c in obs.active_constraints]

    if history_tail:
        lines += ["", "## Recent Conversation (last 4 turns)"]
        lines += [f"  {h}" for h in history_tail]

    lines += [
        "",
        f"Turn {obs.turn_number} of {obs.max_turns} ({turns_remaining} turns remaining)",
        "",
        f'AE: "{obs.ae_message}"',
        "",
        "Respond with valid JSON only:",
        '{"action_type": "counter", "price_per_seat": 0.0, "contract_length": 1.0,',
        ' "annual_increase_cap": 5.0, "message": "..."}',
    ]
    return "\n".join(lines)


def parse_llm_to_action(text: str) -> NegotiateAction:
    """Parse LLM output to NegotiateAction. Never crashes — always returns a valid action."""
    text = text.strip()

    # 1) Try full JSON parse
    try:
        data = json.loads(text)
        valid_keys = set(NegotiateAction.model_fields)
        return NegotiateAction(**{k: v for k, v in data.items() if k in valid_keys})
    except Exception:
        pass

    # 2) Try to extract embedded JSON object
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            valid_keys = set(NegotiateAction.model_fields)
            return NegotiateAction(**{k: v for k, v in data.items() if k in valid_keys})
        except Exception:
            pass

    # 3) Heuristic: scan for action type keyword and numbers
    action_type = "counter"
    for at in ("accept", "walkaway", "probe", "offer", "counter"):
        if at in text.lower():
            action_type = at
            break

    price, length, cap = 0.0, 1.0, 5.0
    dollar = re.search(r"\$\s*([\d]+(?:\.\d+)?)", text)
    if dollar:
        price = float(dollar.group(1))

    nums = re.findall(r"[\d]+(?:\.\d+)?", text)
    if not dollar and nums and action_type in ("offer", "counter"):
        price = float(nums[0])
    if len(nums) >= 2:
        c = float(nums[1])
        if 1.0 <= c <= 5.0:
            length = c
    if len(nums) >= 3:
        c = float(nums[2])
        if 0.0 <= c <= 20.0:
            cap = c

    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=length,
        annual_increase_cap=cap,
        message=text[:200],
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_once(
    trainer: GRPOTrainer,
    env: NegotiateEnvClient,
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int = 10,
) -> dict:
    """Run one full negotiation episode. Returns rollout data for GRPOTrainer."""
    obs = env.reset()

    prompt_ids_all = []
    completion_ids_all = []
    logprobs_all = []
    final_reward = 0.0

    for _turn in range(max_turns):
        if obs.done:
            final_reward = obs.reward
            break

        user_content = format_obs_as_prompt(obs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        # Generate using TRL helper (handles both colocate and server vLLM modes)
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids_all.extend(rollout_outputs["prompt_ids"])
        completion_ids_all.extend(rollout_outputs["completion_ids"])
        logprobs_all.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        action = parse_llm_to_action(completion_text)
        obs = env.step(action)

        if obs.done:
            final_reward = obs.reward
            break

    return {
        "prompt_ids": prompt_ids_all,
        "completion_ids": completion_ids_all,
        "logprobs": logprobs_all,
        "env_reward": final_reward,
    }


def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict:
    """Called by GRPOTrainer once per batch. Runs one episode per prompt."""
    results = [
        rollout_once(
            trainer=trainer,
            env=env,
            tokenizer=tokenizer,
            dataset_prompt=p,
            system_prompt=SYSTEM_PROMPT,
            max_turns=cli_args.max_turns,
        )
        for p in prompts
    ]
    return {
        "prompt_ids":     [r["prompt_ids"]     for r in results],
        "completion_ids": [r["completion_ids"]  for r in results],
        "logprobs":       [r["logprobs"]        for r in results],
        "env_reward":     [r["env_reward"]      for r in results],
    }


# ---------------------------------------------------------------------------
# Reward function (receives env_reward forwarded as kwarg)
# ---------------------------------------------------------------------------

def reward_from_env(completions: list[str], **kwargs) -> list[float]:
    """Extract environment rewards forwarded from rollout_func."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(r) for r in env_rewards]
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global tokenizer, env

    print(f"[NegotiateEnv] Connecting to environment at {cli_args.env_url}")
    env = NegotiateEnvClient(base_url=cli_args.env_url)

    print(f"[NegotiateEnv] Loading tokenizer for {cli_args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cli_args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Each entry triggers one rollout episode; the real context comes from env.reset()
    dataset = Dataset.from_dict({
        "prompt": ["You are a procurement manager. Negotiate the best SaaS contract."]
        * cli_args.num_episodes
    })

    grpo_config = GRPOConfig(
        output_dir=cli_args.output_dir,
        use_vllm=True,
        vllm_mode=cli_args.vllm_mode,
        vllm_server_base_url=cli_args.vllm_server_url,
        num_train_epochs=1,
        num_generations=8,
        max_completion_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=cli_args.model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("[NegotiateEnv] Starting GRPO training ...")
    trainer.train()
    print(f"[NegotiateEnv] Training complete. Model saved to {cli_args.output_dir}")


if __name__ == "__main__":
    main()
