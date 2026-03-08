#!/usr/bin/env python3
"""Evaluation script for NegotiateEnv.

Runs N episodes with a trained LLM agent (or a baseline) and reports:
  - mean / median reward
  - deal success rate
  - average deal price (when deal struck)
  - average negotiation turns
  - strategy frequency (action distribution)

Usage:
    # Evaluate rule-based baseline (no model needed):
    python evaluate.py --agent rule --episodes 100

    # Evaluate random baseline:
    python evaluate.py --agent random --episodes 100

    # Evaluate a trained LLM (requires OPENAI_API_KEY or local model):
    python evaluate.py --agent llm --model gpt-4o-mini --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import Counter
from typing import Callable

from negotiate_env.client.negotiate_env_client import NegotiateEnvClient
from negotiate_env.models import NegotiateAction, NegotiateObservation


# ---------------------------------------------------------------------------
# Agent policies
# ---------------------------------------------------------------------------

def random_policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
    action_type = random.choice(["offer", "counter", "probe", "accept", "walkaway"])
    price = round(random.uniform(obs.current_offer.get("price_per_seat", 100) * 0.5,
                                  obs.current_offer.get("price_per_seat", 100)), 2)
    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=random.choice([1.0, 2.0, 3.0]),
        annual_increase_cap=round(random.uniform(3.0, 8.0), 1),
    )


def rule_policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
    current_price = obs.current_offer.get("price_per_seat", 100.0)
    max_price = obs.your_max_price
    max_length = obs.your_max_length
    max_cap = obs.your_max_cap
    turns_left = obs.max_turns - turn

    if current_price <= max_price:
        return NegotiateAction(action_type="accept", price_per_seat=current_price,
                               contract_length=obs.current_offer.get("contract_length", 2.0),
                               annual_increase_cap=obs.current_offer.get("annual_increase_cap", 7.0))
    if turns_left <= 1:
        return NegotiateAction(action_type="walkaway")
    if turn == 1:
        return NegotiateAction(action_type="probe",
                               message="What's your best price on a multi-year deal?")
    target = max(max_price, current_price * 0.93)
    return NegotiateAction(action_type="counter", price_per_seat=round(target, 2),
                           contract_length=min(2.0, max_length), annual_increase_cap=max_cap)


def make_llm_policy(model: str) -> Callable:
    """Returns a policy function that calls an OpenAI-compatible model."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    SYSTEM = (
        "You are a procurement manager negotiating a B2B SaaS contract. "
        "Respond with a JSON action: "
        '{"action_type": "counter", "price_per_seat": 85.0, '
        '"contract_length": 1.0, "annual_increase_cap": 3.0, "message": "..."}'
    )

    def policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
        from negotiate_env.client.negotiate_env_client import observation_to_prompt
        prompt = observation_to_prompt(obs)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return NegotiateEnvClient.parse_llm_output_to_action(resp.choices[0].message.content)

    return policy


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: NegotiateEnvClient,
    policy: Callable,
    max_turns: int,
) -> dict:
    obs = env.reset(max_turns=max_turns)
    turn = 0
    action_counts: Counter = Counter()

    while not obs.done:
        turn += 1
        action = policy(obs, turn)
        action_counts[action.action_type] += 1
        obs = env.step(action)

    deal_price = obs.current_offer.get("price_per_seat", 0.0) if obs.reward > 0 else None
    return {
        "reward": obs.reward,
        "success": obs.reward > 0,
        "turns": turn,
        "deal_price": deal_price,
        "action_counts": dict(action_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "rule", "llm"], default="rule")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name (for --agent llm)")
    parser.add_argument("--env-url", default="http://127.0.0.1:7860")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    env = NegotiateEnvClient(base_url=args.env_url)

    if args.agent == "random":
        policy = random_policy
    elif args.agent == "rule":
        policy = rule_policy
    else:
        policy = make_llm_policy(args.model)

    results = []
    total_action_counts: Counter = Counter()

    print(f"Evaluating '{args.agent}' agent over {args.episodes} episodes "
          f"(difficulty={args.difficulty})...")

    for i in range(args.episodes):
        ep = run_episode(env, policy, args.max_turns)
        results.append(ep)
        total_action_counts.update(ep["action_counts"])
        if (i + 1) % 20 == 0:
            running_mean = statistics.mean(r["reward"] for r in results)
            print(f"  Episode {i+1:>4}: running mean reward = {running_mean:.4f}")

    rewards = [r["reward"] for r in results]
    successes = [r for r in results if r["success"]]
    deal_prices = [r["deal_price"] for r in successes if r["deal_price"] is not None]
    turns_list = [r["turns"] for r in results]

    print(f"\n{'='*50}")
    print(f"  Agent:           {args.agent}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Mean reward:     {statistics.mean(rewards):.4f}")
    print(f"  Median reward:   {statistics.median(rewards):.4f}")
    print(f"  Std reward:      {statistics.stdev(rewards):.4f}" if len(rewards) > 1 else "")
    print(f"  Success rate:    {len(successes) / args.episodes:.1%}")
    print(f"  Avg deal price:  ${statistics.mean(deal_prices):.2f}" if deal_prices else "  Avg deal price:  N/A")
    print(f"  Avg turns:       {statistics.mean(turns_list):.1f}")
    print(f"\n  Strategy distribution:")
    total_actions = sum(total_action_counts.values()) or 1
    for action_type in ["counter", "probe", "offer", "accept", "walkaway"]:
        count = total_action_counts.get(action_type, 0)
        pct = count / total_actions
        print(f"    {action_type:<12} {count:>5}  ({pct:.1%})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
