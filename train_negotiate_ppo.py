#!/usr/bin/env python3
"""
NegotiateEnv PPO Training Script — Unsloth edition
Trains Qwen/Qwen2.5-1.5B-Instruct with 4-bit LoRA to negotiate B2B SaaS contracts.
"""

import argparse
import json
import re
import time
import requests
import torch
from datasets import Dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# CLI
parser = argparse.ArgumentParser(description="NegotiateEnv PPO training with Unsloth")
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--env-url", default="http://localhost:7860")
parser.add_argument("--max-turns", type=int, default=10)
parser.add_argument("--num-episodes", type=int, default=700)
parser.add_argument("--output-dir", default="negotiate-ppo-output")
parser.add_argument("--lora-rank", type=int, default=16)
parser.add_argument("--max-seq-length", type=int, default=1024)
cli_args = parser.parse_args()

model = None
tokenizer = None

SYSTEM_PROMPT = """You are an expert procurement manager negotiating a B2B SaaS contract.
Your goal is to get the best deal: lowest price per seat, shortest contract length,
and lowest annual price increase cap.

Respond with a single JSON action:
{"action_type": "counter", "price_per_seat": 85.0, "contract_length": 1.0,
 "annual_increase_cap": 3.0, "message": "Your message to the AE"}

action_type must be one of: "offer", "counter", "probe", "accept", "walkaway"
Always respond with valid JSON only."""


def env_reset(env_url: str, scenario_id: str | None = None) -> dict:
    payload = {}
    if scenario_id:
        payload["scenario_id"] = scenario_id
    r = requests.post(f"{env_url}/reset", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def env_step(env_url: str, action: dict) -> dict:
    r = requests.post(f"{env_url}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def obs_to_prompt(obs: dict) -> str:
    cur = obs.get("current_offer", {})
    hist = obs.get("conversation_history", [])[-4:]
    turns_left = obs.get("max_turns", 10) - obs.get("turn_number", 0)

    lines = [
        "## Negotiation Context",
        obs.get("context", ""),
        "",
        "## AE's Current Offer",
        f"  Price:    ${cur.get('price_per_seat', 0):.2f}/seat/month",
        f"  Length:   {cur.get('contract_length', 0):.0f} year(s)",
        f"  Ann. cap: {cur.get('annual_increase_cap', 0):.1f}%",
        "",
        "## Your Budget Limits",
        f"  Max price:  ${obs.get('your_max_price', 0):.2f}/seat/month",
        f"  Max length: {obs.get('your_max_length', 0):.0f} year(s)",
        f"  Max cap:    {obs.get('your_max_cap', 0):.1f}%",
    ]

    constraints = obs.get("active_constraints", [])
    if constraints:
        lines += ["", "## Active Constraints"]
        lines += [f"  - {c}" for c in constraints]

    if hist:
        lines += ["", "## Recent Conversation"]
        lines += [f"  {h}" for h in hist]

    lines += [
        "",
        f"Turn {obs.get('turn_number', 0)} of {obs.get('max_turns', 10)} "
        f"({turns_left} remaining)",
        "",
        f'AE: "{obs.get("ae_message", "")}"',
        "",
        "Respond with valid JSON only:",
        '{"action_type": "counter", "price_per_seat": 0.0, '
        '"contract_length": 1.0, "annual_increase_cap": 5.0, "message": "..."}',
    ]
    return "\n".join(lines)


def parse_to_action(text: str) -> dict:
    text = text.strip()
    valid_types = {"offer", "counter", "probe", "accept", "walkaway"}

    try:
        data = json.loads(text)
        if data.get("action_type") in valid_types:
            return data
    except Exception:
        pass

    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if data.get("action_type") in valid_types:
                return data
        except Exception:
            pass

    action_type = "counter"
    for at in ("accept", "walkaway", "probe", "offer", "counter"):
        if at in text.lower():
            action_type = at
            break

    price = 0.0
    dollar = re.search(r"\$\s*([\d]+(?:\.\d+)?)", text)
    if dollar:
        price = float(dollar.group(1))

    return {
        "action_type": action_type,
        "price_per_seat": price,
        "contract_length": 1.0,
        "annual_increase_cap": 5.0,
        "message": text[:200],
    }


def run_episode(prompt: str) -> tuple[str, float]:
    """Run a full episode and return (response, reward)"""
    if model is None or tokenizer is None:
        return "", 0.0

    max_retries = 3
    retry_delay = 2

    try:
        # Reset with retries
        for attempt in range(max_retries):
            try:
                obs = env_reset(cli_args.env_url)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise

        if obs.get("done", False):
            return "", 0.0

        # Generate first action
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_to_prompt(obs)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=cli_args.max_seq_length - 256,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        action = parse_to_action(response)
        
        # Step with retries
        for attempt in range(max_retries):
            try:
                obs = env_step(cli_args.env_url, action)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise

        if obs.get("done", False):
            return response, float(obs.get("reward", 0.0))

        # Continue for remaining turns
        for _ in range(1, cli_args.max_turns):
            if obs.get("done", False):
                break

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_to_prompt(obs)},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=cli_args.max_seq_length - 256,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            action = parse_to_action(completion)
            
            # Step with retries
            for attempt in range(max_retries):
                try:
                    obs = env_step(cli_args.env_url, action)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise

        return response, float(obs.get("reward", 0.0))

    except Exception as e:
        print(f"[warn] Episode failed: {e}")
        return "", 0.0


def build_dataset(env_url: str, n: int, tokenizer) -> Dataset:
    prompts = []
    for _ in range(n):
        try:
            obs = env_reset(env_url)
            user_content = obs_to_prompt(obs)
        except Exception:
            user_content = "Negotiate a B2B SaaS contract. Respond with a JSON action."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt_text)

    return Dataset.from_dict({"query": prompts})


def main() -> None:
    global model, tokenizer

    print(f"[NegotiateEnv/PPO] Loading {cli_args.model_id}...")
    
    # Load with Unsloth
    from unsloth import FastLanguageModel

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cli_args.model_id,
        max_seq_length=cli_args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add LoRA
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=cli_args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cli_args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Wrap with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    print(f"[NegotiateEnv/PPO] Building dataset ({cli_args.num_episodes} episodes)...")
    dataset = build_dataset(cli_args.env_url, cli_args.num_episodes, tokenizer)

    # PPO config
    ppo_config = PPOConfig(
        model_name=cli_args.model_id,
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        log_with=None,
    )

    # Trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("[NegotiateEnv/PPO] Starting training...")
    
    # Training loop
    for epoch in range(1):
        for batch_idx, batch in enumerate(trainer.dataloader):
            query_tensors = batch["input_ids"]
            
            # Generate responses
            response_tensors = trainer.generate(
                query_tensors,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decode and get rewards
            batch_rewards = []
            for query_tensor, response_tensor in zip(query_tensors, response_tensors):
                response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
                _, reward = run_episode(response_text)
                batch_rewards.append(torch.tensor(reward))
            
            # PPO step
            stats = trainer.step(query_tensors, response_tensors, batch_rewards)
            
            if batch_idx % 10 == 0:
                mean_reward = sum(r.item() for r in batch_rewards) / len(batch_rewards)
                print(f"Batch {batch_idx}: mean_reward={mean_reward:.4f}")

    # Save
    model.save_pretrained(cli_args.output_dir)
    tokenizer.save_pretrained(cli_args.output_dir)
    print(f"[NegotiateEnv/PPO] Done. Model saved to {cli_args.output_dir}/")


if __name__ == "__main__":
    main()
