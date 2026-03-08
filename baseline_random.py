#!/usr/bin/env python3
"""Random baseline agent for NegotiateEnv.

Selects actions uniformly at random. Used as a lower-bound baseline.

Usage:
    # Start server first: uvicorn negotiate_env.server.app:app --port 7860
    python baseline_random.py --episodes 50
"""

import argparse
import random
import statistics

from negotiate_env.client.negotiate_env_client import NegotiateEnvClient
from negotiate_env.models import NegotiateAction

ACTION_TYPES = ["offer", "counter", "probe", "accept", "walkaway"]


def random_action(obs) -> NegotiateAction:
    action_type = random.choice(ACTION_TYPES)
    current_price = obs.current_offer.get("price_per_seat", 100.0)
    # Random price between 50% and 100% of current offer
    price = round(random.uniform(current_price * 0.5, current_price), 2)
    length = random.choice([1.0, 2.0, 3.0])
    cap = round(random.uniform(3.0, 8.0), 1)
    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=length,
        annual_increase_cap=cap,
        message="Random agent action.",
    )


def run_episode(env: NegotiateEnvClient, max_turns: int = 10) -> float:
    obs = env.reset(max_turns=max_turns)
    while not obs.done:
        action = random_action(obs)
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

    print(f"Running {args.episodes} random-agent episodes...")
    for i in range(args.episodes):
        r = run_episode(env, max_turns=args.max_turns)
        rewards.append(r)
        if r > 0:
            successes += 1
        if (i + 1) % 10 == 0:
            print(f"  Episode {i+1}: reward={r:.4f}")

    print("\n=== Random Agent Results ===")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Mean reward:   {statistics.mean(rewards):.4f}")
    print(f"  Median reward: {statistics.median(rewards):.4f}")
    print(f"  Success rate:  {successes / args.episodes:.1%}")


if __name__ == "__main__":
    main()
