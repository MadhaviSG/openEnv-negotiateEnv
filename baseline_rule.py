#!/usr/bin/env python3
"""Rule-based baseline agent for NegotiateEnv.

Strategy:
  1. Turn 1: probe to gather information.
  2. Turn 2: counter at competitor_price (if known) or 85% of list price.
  3. Turn 3+: extend contract length to 2 years to unlock concessions.
  4. Accept if vendor price <= agent max price.
  5. Walk away if no deal by turn max_turns - 1.

Usage:
    # Start server first: uvicorn negotiate_env.server.app:app --port 7860
    python baseline_rule.py --episodes 50
"""

import argparse
import statistics

from negotiate_env.client.negotiate_env_client import NegotiateEnvClient
from negotiate_env.models import NegotiateAction


def rule_action(obs, turn: int) -> NegotiateAction:
    current_price = obs.current_offer.get("price_per_seat", 100.0)
    current_length = obs.current_offer.get("contract_length", 2.0)
    current_cap = obs.current_offer.get("annual_increase_cap", 7.0)
    max_price = obs.your_max_price
    max_length = obs.your_max_length
    max_cap = obs.your_max_cap
    turns_left = obs.max_turns - turn

    # Accept if current offer is within budget
    if current_price <= max_price and current_length <= max_length and current_cap <= max_cap:
        return NegotiateAction(
            action_type="accept",
            price_per_seat=current_price,
            contract_length=current_length,
            annual_increase_cap=current_cap,
            message="This works within our budget. Let's proceed.",
        )

    # Walk away if almost out of turns and still over budget
    if turns_left <= 1:
        return NegotiateAction(
            action_type="walkaway",
            message="We can't reach an agreement within our constraints.",
        )

    # Turn 1: probe
    if turn == 1:
        return NegotiateAction(
            action_type="probe",
            message="What's the best you can do on price if we commit to a multi-year deal?",
        )

    # Turn 2: counter at max_price with 2-year term
    if turn == 2:
        return NegotiateAction(
            action_type="counter",
            price_per_seat=max_price,
            contract_length=min(2.0, max_length),
            annual_increase_cap=max_cap,
            message=f"Our budget is ${max_price:.0f}/seat. We can do a 2-year with a {max_cap:.0f}% cap.",
        )

    # Turn 3+: try extending contract to unlock more discount
    if turn == 3 and max_length >= 3.0:
        return NegotiateAction(
            action_type="counter",
            price_per_seat=max_price,
            contract_length=3.0,
            annual_increase_cap=max_cap,
            message=f"We'll commit to 3 years if you can meet ${max_price:.0f}/seat.",
        )

    # Otherwise: nudge price down by 5% each turn
    target_price = max(max_price * 0.9, current_price * 0.95)
    return NegotiateAction(
        action_type="counter",
        price_per_seat=round(target_price, 2),
        contract_length=min(2.0, max_length),
        annual_increase_cap=max_cap,
        message=f"We need to be at ${target_price:.0f}/seat to make this work.",
    )


def run_episode(env: NegotiateEnvClient, max_turns: int = 10) -> float:
    obs = env.reset(max_turns=max_turns)
    turn = 0
    while not obs.done:
        turn += 1
        action = rule_action(obs, turn)
        obs = env.step(action)
    return obs.reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:7860")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    env = NegotiateEnvClient(base_url=args.env_url)
    rewards = []
    successes = 0

    print(f"Running {args.episodes} rule-based agent episodes...")
    for i in range(args.episodes):
        r = run_episode(env, max_turns=args.max_turns)
        rewards.append(r)
        if r > 0:
            successes += 1
        if (i + 1) % 10 == 0:
            print(f"  Episode {i+1}: reward={r:.4f}")

    print("\n=== Rule-Based Agent Results ===")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Mean reward:   {statistics.mean(rewards):.4f}")
    print(f"  Median reward: {statistics.median(rewards):.4f}")
    print(f"  Success rate:  {successes / args.episodes:.1%}")


if __name__ == "__main__":
    main()
