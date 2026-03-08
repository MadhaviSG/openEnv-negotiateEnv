#!/usr/bin/env python3
"""Task 40 — Compare strategy (action) distribution before vs after training.

Runs evaluate.py-style episodes against the environment using two agents
(baseline rule-based = "before", trained LLM = "after") and plots a
side-by-side bar chart of action type frequencies.

Usage:
    # Compare rule baseline vs trained LLM (requires running env server):
    python plot_strategy_distribution.py \
        --before-agent rule \
        --after-agent llm \
        --after-model gpt-4o-mini \
        --env-url http://127.0.0.1:7860 \
        --episodes 100

    # Compare two saved JSON result files (no live env needed):
    python plot_strategy_distribution.py \
        --before-json results_before.json \
        --after-json results_after.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

ACTION_TYPES = ["counter", "offer", "probe", "accept", "walkaway"]


# ---------------------------------------------------------------------------
# Lightweight episode runner (mirrors evaluate.py logic, no import needed)
# ---------------------------------------------------------------------------

def _run_episodes_rule(env_url: str, n: int, max_turns: int) -> dict[str, int]:
    """Run rule-based agent, return aggregated action counts."""
    from negotiate_env.client.negotiate_env_client import NegotiateEnvClient
    from negotiate_env.models import NegotiateAction

    env = NegotiateEnvClient(base_url=env_url)
    counts: Counter = Counter()

    for _ in range(n):
        obs = env.reset(max_turns=max_turns)
        turn = 0
        while not obs.done:
            turn += 1
            cur_price = obs.current_offer.get("price_per_seat", 100.0)
            turns_left = obs.max_turns - turn
            if cur_price <= obs.your_max_price:
                action = NegotiateAction(action_type="accept", price_per_seat=cur_price,
                                         contract_length=obs.current_offer.get("contract_length", 2.0),
                                         annual_increase_cap=obs.current_offer.get("annual_increase_cap", 7.0))
            elif turns_left <= 1:
                action = NegotiateAction(action_type="walkaway")
            elif turn == 1:
                action = NegotiateAction(action_type="probe",
                                          message="What's your best price on a multi-year deal?")
            else:
                target = max(obs.your_max_price, cur_price * 0.93)
                action = NegotiateAction(action_type="counter", price_per_seat=round(target, 2),
                                          contract_length=min(2.0, obs.your_max_length),
                                          annual_increase_cap=obs.your_max_cap)
            counts[action.action_type] += 1
            obs = env.step(action)

    return dict(counts)


def _run_episodes_llm(env_url: str, model: str, n: int, max_turns: int) -> dict[str, int]:
    """Run LLM agent, return aggregated action counts."""
    import os
    from openai import OpenAI
    from negotiate_env.client.negotiate_env_client import NegotiateEnvClient, observation_to_prompt
    from negotiate_env.models import NegotiateAction

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    env = NegotiateEnvClient(base_url=env_url)
    counts: Counter = Counter()

    SYSTEM = (
        "You are a procurement manager negotiating a B2B SaaS contract. "
        'Respond with JSON: {"action_type": "counter", "price_per_seat": 85.0, '
        '"contract_length": 1.0, "annual_increase_cap": 3.0, "message": "..."}'
    )

    for _ in range(n):
        obs = env.reset(max_turns=max_turns)
        while not obs.done:
            prompt = observation_to_prompt(obs)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
            )
            action = NegotiateEnvClient.parse_llm_output_to_action(resp.choices[0].message.content)
            counts[action.action_type] += 1
            obs = env.step(action)

    return dict(counts)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    before_label: str,
    after_label: str,
    out_path: str,
) -> None:
    def to_pct(counts: dict[str, int]) -> list[float]:
        total = sum(counts.values()) or 1
        return [counts.get(a, 0) / total * 100 for a in ACTION_TYPES]

    before_pct = to_pct(before_counts)
    after_pct = to_pct(after_counts)

    x = np.arange(len(ACTION_TYPES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_before = ax.bar(x - width / 2, before_pct, width, label=before_label, color="steelblue", alpha=0.85)
    bars_after = ax.bar(x + width / 2, after_pct, width, label=after_label, color="coral", alpha=0.85)

    ax.set_xlabel("Action Type")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("NegotiateEnv — Strategy Distribution: Before vs After Training")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_TYPES)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars with percentage
    for bar in (*bars_before, *bars_after):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot_strategy_distribution] Saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-agent", choices=["rule", "random"], default="rule")
    parser.add_argument("--after-agent", choices=["llm", "rule", "random"], default="llm")
    parser.add_argument("--after-model", default="gpt-4o-mini")
    parser.add_argument("--env-url", default="http://127.0.0.1:7860")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--before-json", default=None, help="Pre-computed before counts JSON")
    parser.add_argument("--after-json", default=None, help="Pre-computed after counts JSON")
    parser.add_argument("--out", default="strategy_distribution.png")
    args = parser.parse_args()

    if args.before_json:
        with open(args.before_json) as f:
            before_counts = json.load(f)
        before_label = "Before (loaded)"
    else:
        print(f"[plot_strategy_distribution] Running {args.before_agent} baseline ({args.episodes} eps)...")
        before_counts = _run_episodes_rule(args.env_url, args.episodes, args.max_turns)
        before_label = f"Before ({args.before_agent})"

    if args.after_json:
        with open(args.after_json) as f:
            after_counts = json.load(f)
        after_label = "After (loaded)"
    else:
        print(f"[plot_strategy_distribution] Running {args.after_agent} agent ({args.episodes} eps)...")
        if args.after_agent == "llm":
            after_counts = _run_episodes_llm(args.env_url, args.after_model, args.episodes, args.max_turns)
        else:
            after_counts = _run_episodes_rule(args.env_url, args.episodes, args.max_turns)
        after_label = f"After ({args.after_agent})"

    print(f"  Before: {before_counts}")
    print(f"  After:  {after_counts}")
    plot_comparison(before_counts, after_counts, before_label, after_label, args.out)


if __name__ == "__main__":
    main()
