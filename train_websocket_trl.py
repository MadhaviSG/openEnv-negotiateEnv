#!/usr/bin/env python3
"""
NegotiateEnv Training with WebSocket (Proper OpenEnv Protocol)
Uses WebSocket for reliable session management.
"""

import argparse
import json
import re
import asyncio
import websockets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--env-url", default="wss://kushaladhyaru-negotiate-env.hf.space")
parser.add_argument("--num-episodes", type=int, default=500)
parser.add_argument("--output-dir", default="negotiate-trl-output")
parser.add_argument("--max-turns", type=int, default=10)
args = parser.parse_args()

SYSTEM_PROMPT = """You are an expert procurement manager negotiating a B2B SaaS contract.
Your goal is to get the best deal: lowest price per seat, shortest contract length,
and lowest annual price increase cap.

Respond with a single JSON action:
{"action_type": "counter", "price_per_seat": 85.0, "contract_length": 1.0,
 "annual_increase_cap": 3.0, "message": "Your message to the AE"}

action_type must be one of: "offer", "counter", "probe", "accept", "walkaway"
Always respond with valid JSON only."""


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

    if obs.get("active_constraints"):
        lines += ["", "## Active Constraints"]
        lines += [f"  - {c}" for c in obs["active_constraints"]]

    if hist:
        lines += ["", "## Recent Conversation"]
        lines += [f"  {h}" for h in hist]

    lines += [
        "",
        f"Turn {obs.get('turn_number', 0)} of {obs.get('max_turns', 10)} ({turns_left} remaining)",
        "",
        f'AE: "{obs.get("ae_message", "")}"',
        "",
        "Respond with valid JSON only:",
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


async def run_episode_ws(websocket, model, tokenizer) -> tuple[list[str], list[str], float]:
    """Run one episode via WebSocket."""
    prompts = []
    responses = []
    
    # Reset
    await websocket.send(json.dumps({"type": "reset"}))
    response = json.loads(await websocket.recv())
    obs = response.get("observation", response)
    
    for turn in range(args.max_turns):
        if obs.get("done", False):
            break
        
        # Create prompt
        user_content = obs_to_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Generate response
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # Store for training
        prompts.append(prompt_text)
        responses.append(response_text)
        
        # Take action
        action = parse_to_action(response_text)
        await websocket.send(json.dumps({"type": "step", "action": action}))
        response = json.loads(await websocket.recv())
        obs = response.get("observation", response)
    
    reward = obs.get("reward", 0.0)
    return prompts, responses, reward


async def collect_episodes(num_episodes: int, model, tokenizer) -> list[dict]:
    """Collect training data via WebSocket."""
    print(f"Collecting {num_episodes} episodes via WebSocket...")
    
    training_examples = []
    total_reward = 0.0
    
    # Convert http/https to ws/wss
    ws_url = args.env_url.replace("https://", "wss://").replace("http://", "ws://")
    if not ws_url.startswith("ws"):
        ws_url = f"wss://{ws_url}"
    
    for ep in range(num_episodes):
        try:
            async with websockets.connect(ws_url, timeout=30) as websocket:
                prompts, responses, reward = await run_episode_ws(websocket, model, tokenizer)
                total_reward += reward
                
                # Weight examples by episode reward
                weight = max(0.1, reward)
                
                for prompt, response in zip(prompts, responses):
                    training_examples.append({
                        "text": prompt + response,
                        "weight": weight,
                    })
                
                if (ep + 1) % 50 == 0:
                    avg_reward = total_reward / (ep + 1)
                    print(f"Episode {ep+1}/{num_episodes}: avg_reward={avg_reward:.4f}")
        
        except Exception as e:
            print(f"[warn] Episode {ep+1} failed: {e}")
            continue
    
    print(f"\nCollected {len(training_examples)} training examples")
    print(f"Average episode reward: {total_reward/num_episodes:.4f}")
    
    return training_examples


def main():
    print(f"[NegotiateEnv/WebSocket] Training with WebSocket protocol")
    print(f"Model: {args.model_id}")
    print(f"Environment: {args.env_url}")
    print(f"Episodes: {args.num_episodes}")
    
    # Load model for data collection
    print("\nLoading model for data collection...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Collect training data
    training_data = asyncio.run(collect_episodes(args.num_episodes, model, tokenizer))
    dataset = Dataset.from_list(training_data)
    
    # Reload model for training
    print("\nLoading model for training...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
