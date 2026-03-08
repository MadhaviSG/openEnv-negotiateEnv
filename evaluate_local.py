#!/usr/bin/env python3
"""Local evaluation script (no network calls)."""

import argparse
import random
import statistics
from collections import Counter

from negotiate_env.server.environment import NegotiateEnvironment
from negotiate_env.models import NegotiateAction

parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="rule", choices=["rule", "random"])
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--max-turns", type=int, default=10)
args = parser.parse_args()


def random_policy(obs, turn: int) -> NegotiateAction:
    action_type = random.choice(["offer", "counter", "probe", "accept", "walkaway"])
    price = round(random.uniform(obs.current_offer.get("price_per_seat", 100) * 0.5,
                                  obs.current_offer.get("price_per_seat", 100)), 2)
    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=random.choice([1.0, 2.0, 3.0]),
        annual_increase_cap=round(random.uniform(3.0, 8.0), 1),
        message="",
    )


def rule_policy(obs, turn: int) -> NegotiateAction:
    """Simple rule-based policy."""
    cur = obs.current_offer
    
    # Probe first
    if turn == 0:
        return NegotiateAction(
            action_type="probe",
            price_per_seat=0.0,
            contract_length=0.0,
            annual_increase_cap=0.0,
            message="Can you tell me more about your pricing?",
        )
    
    # Counter with budget-aware offer
    target_price = obs.your_max_price * 0.95
    target_length = min(obs.your_max_length, 2.0)
    target_cap = obs.your_max_cap
    
    return NegotiateAction(
        action_type="counter",
        price_per_seat=target_price,
        contract_length=target_length,
        annual_increase_cap=target_cap,
        message=f"I can do ${target_price:.0f}/seat for {target_length:.0f} years with {target_cap:.0f}% cap",
    )


def run_episode(env: NegotiateEnvironment, policy, max_turns: int):
    """Run one episode."""
    obs = env.reset()
    actions = []
    
    for turn in range(max_turns):
        if obs.done:
            break
        
        action = policy(obs, turn)
        actions.append(action.action_type)
        obs = env.step(action)
    
    return obs.reward, obs.done, actions


def main():
    print(f"Evaluating '{args.agent}' agent over {args.episodes} episodes (difficulty=medium)...")
    
    # Select policy
    if args.agent == "rule":
        policy = rule_policy
    else:
        policy = random_policy
    
    # Run episodes
    env = NegotiateEnvironment(difficulty="medium", use_hf_dataset=True)
    
    rewards = []
    successes = 0
    action_counter = Counter()
    
    for ep in range(args.episodes):
        reward, done, actions = run_episode(env, policy, args.max_turns)
        rewards.append(reward)
        if done:
            successes += 1
        action_counter.update(actions)
        
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:3d}: running mean reward = {sum(rewards)/len(rewards):.4f}")
    
    # Results
    print("\n" + "="*50)
    print(f"  Agent:           {args.agent}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Mean reward:     {sum(rewards)/len(rewards):.4f}")
    print(f"  Median reward:   {statistics.median(rewards):.4f}")
    print(f"  Std reward:      {statistics.stdev(rewards):.4f}")
    print(f"  Success rate:    {successes/args.episodes*100:.1f}%")
    print(f"\n  Strategy distribution:")
    for action_type, count in action_counter.most_common():
        pct = count / sum(action_counter.values()) * 100
        print(f"    {action_type:12s} {count:3d}  ({pct:.1f}%)")
    print("="*50)


if __name__ == "__main__":
    main()
