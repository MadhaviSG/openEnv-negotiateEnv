#!/usr/bin/env python3
"""
NegotiateEnv Training with Pure TRL (No Unsloth, No HTTP)
Uses local environment instance - no network calls needed.
"""

import argparse
import json
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# Local environment import
from negotiate_env.server.environment import NegotiateEnvironment
from negotiate_env.models import NegotiateAction

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
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


def obs_to_prompt(obs) -> str:
    cur = obs.current_offer
    hist = obs.conversation_history[-4:]
    turns_left = obs.max_turns - obs.turn_number

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
    ]

    if obs.active_constraints:
        lines += ["", "## Active Constraints"]
        lines += [f"  - {c}" for c in obs.active_constraints]

    if hist:
        lines += ["", "## Recent Conversation"]
        lines += [f"  {h}" for h in hist]

    lines += [
        "",
        f"Turn {obs.turn_number} of {obs.max_turns} ({turns_left} remaining)",
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
    """Run one episode and collect (prompt, response) pairs + final reward."""
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
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # Store for training
        prompts.append(prompt_text)
        responses.append(response)
        
        # Take action
        action = parse_to_action(response)
        obs = env.step(action)
    
    return prompts, responses, obs.reward


def build_training_data(num_episodes: int) -> list[dict]:
    """Collect episodes and build training dataset."""
    print(f"Collecting {num_episodes} episodes...")
    
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
        
        # Weight examples by episode reward (higher reward = more weight)
        weight = max(0.1, reward)  # Minimum weight 0.1
        
        for prompt, response in zip(prompts, responses):
            training_examples.append({
                "text": prompt + response,
                "weight": weight,
            })
        
        if (ep + 1) % 50 == 0:
            avg_reward = total_reward / (ep + 1)
            print(f"Episode {ep+1}/{num_episodes}: avg_reward={avg_reward:.4f}")
    
    print(f"\nCollected {len(training_examples)} training examples")
    print(f"Average episode reward: {total_reward/num_episodes:.4f}")
    
    return training_examples


def main():
    print(f"[NegotiateEnv/TRL] Training with local environment")
    print(f"Model: {args.model_id}")
    print(f"Episodes: {args.num_episodes}")
    
    # Collect training data
    training_data = build_training_data(args.num_episodes)
    dataset = Dataset.from_list(training_data)
    
    # Load model and tokenizer
    print("\nLoading model for training...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
        processing_class=tokenizer,  # Changed from tokenizer to processing_class
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
