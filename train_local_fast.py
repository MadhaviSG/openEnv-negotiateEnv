#!/usr/bin/env python3
"""
Fast local training - optimized for speed.
Reduces episodes and uses simpler training.
"""

import argparse
import json
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

from negotiate_env.server.environment import NegotiateEnvironment
from negotiate_env.models import NegotiateAction

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--num-episodes", type=int, default=200)  # Reduced from 500
parser.add_argument("--output-dir", default="negotiate-trl-output")
parser.add_argument("--max-turns", type=int, default=6)  # Reduced from 10
args = parser.parse_args()

SYSTEM_PROMPT = """You are an expert procurement manager negotiating a B2B SaaS contract.
Your goal is to get the best deal: lowest price per seat, shortest contract length,
and lowest annual price increase cap.

Respond with a single JSON action:
{"action_type": "counter", "price_per_seat": 85.0, "contract_length": 1.0,
 "annual_increase_cap": 3.0, "message": "Your message to the AE"}

action_type must be one of: "offer", "counter", "probe", "accept", "walkaway"
Always respond with valid JSON only."""


def obs_to_prompt(obs) -> str:
    cur = obs.current_offer
    lines = [
        "## Negotiation Context",
        obs.context,
        "",
        "## AE's Current Offer",
        f"  Price:    ${cur['price_per_seat']:.2f}/seat/month",
        f"  Length:   {cur['contract_length']:.0f} year(s)",
        f"  Ann. cap: {cur['annual_increase_cap']:.1f}%",
        "",
        "## Your Budget Limits",
        f"  Max price:  ${obs.your_max_price:.2f}/seat/month",
        f"  Max length: {obs.your_max_length:.0f} year(s)",
        f"  Max cap:    {obs.your_max_cap:.1f}%",
        "",
        f'AE: "{obs.ae_message}"',
        "",
        "Respond with valid JSON only:",
    ]
    return "\n".join(lines)


def parse_to_action(text: str) -> NegotiateAction:
    text = text.strip()
    valid_types = {"offer", "counter", "probe", "accept", "walkaway"}

    try:
        data = json.loads(text)
        if data.get("action_type") in valid_types:
            return NegotiateAction(**data)
    except Exception:
        pass

    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if data.get("action_type") in valid_types:
                return NegotiateAction(**data)
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

    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=1.0,
        annual_increase_cap=5.0,
        message=text[:200],
    )


def run_episode(env: NegotiateEnvironment, model, tokenizer) -> tuple[list[str], list[str], float]:
    """Run one episode - simplified for speed."""
    prompts = []
    responses = []
    
    obs = env.reset()
    
    for turn in range(args.max_turns):
        if obs.done:
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
        
        # Generate response (faster settings)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced from 256
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # Store for training
        prompts.append(prompt_text)
        responses.append(response)
        
        # Take action
        action = parse_to_action(response)
        obs = env.step(action)
    
    return prompts, responses, obs.reward


def build_training_data(num_episodes: int) -> list[dict]:
    """Collect episodes - optimized for speed."""
    print(f"Collecting {num_episodes} episodes (fast mode)...")
    
    # Load model for data collection
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    env = NegotiateEnvironment(difficulty="medium", use_hf_dataset=True)
    
    training_examples = []
    total_reward = 0.0
    
    for ep in range(num_episodes):
        prompts, responses, reward = run_episode(env, model, tokenizer)
        total_reward += reward
        
        # Weight examples by episode reward
        weight = max(0.1, reward)
        
        for prompt, response in zip(prompts, responses):
            training_examples.append({
                "text": prompt + response,
                "weight": weight,
            })
        
        if (ep + 1) % 25 == 0:  # Report every 25 episodes
            avg_reward = total_reward / (ep + 1)
            print(f"Episode {ep+1}/{num_episodes}: avg_reward={avg_reward:.4f}")
    
    print(f"\nCollected {len(training_examples)} training examples")
    print(f"Average episode reward: {total_reward/num_episodes:.4f}")
    
    return training_examples


def main():
    print(f"[Fast Training] Local environment")
    print(f"Model: {args.model_id}")
    print(f"Episodes: {args.num_episodes} (reduced for speed)")
    print(f"Max turns: {args.max_turns} (reduced for speed)")
    
    # Collect training data
    training_data = build_training_data(args.num_episodes)
    dataset = Dataset.from_list(training_data)
    
    # Load model for training
    print("\nLoading model for training...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add LoRA (smaller for speed)
    lora_config = LoraConfig(
        r=8,  # Reduced from 16
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Fewer modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Training arguments (faster)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,  # Reduced from 3
        per_device_train_batch_size=8,  # Increased from 4
        gradient_accumulation_steps=2,  # Reduced from 4
        learning_rate=2e-5,
        bf16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=1,
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
