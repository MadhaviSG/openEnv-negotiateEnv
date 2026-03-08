#!/usr/bin/env python3
"""Task 39 — Plot training reward curve from TRL trainer log history.

Usage:
    # After training completes, pass the trainer_state.json saved in output dir:
    python plot_reward_curve.py --log-file negotiate-grpo-output/trainer_state.json

    # Or pipe a JSON list of log entries directly:
    python plot_reward_curve.py --log-file training_log.json --out reward_curve.png
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt


def load_log_history(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    # trainer_state.json wraps entries under "log_history"
    if isinstance(data, dict) and "log_history" in data:
        return data["log_history"]
    # plain list of log dicts
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised log format in {path}")


def extract_reward_series(log_history: list[dict]) -> tuple[list[int], list[float]]:
    steps, rewards = [], []
    for entry in log_history:
        # TRL logs env reward under various keys depending on version
        reward = (
            entry.get("env_reward")
            or entry.get("reward")
            or entry.get("train/env_reward")
            or entry.get("train/reward")
        )
        step = entry.get("step") or entry.get("epoch")
        if reward is not None and step is not None:
            steps.append(float(step))
            rewards.append(float(reward))
    return steps, rewards


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Simple moving average."""
    out = []
    for i, v in enumerate(values):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def plot(steps: list[float], rewards: list[float], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, alpha=0.3, color="steelblue", linewidth=1, label="raw")
    if len(rewards) >= 5:
        ax.plot(steps, smooth(rewards), color="steelblue", linewidth=2, label="smoothed (MA-10)")
    ax.set_xlabel("Training Step / Epoch")
    ax.set_ylabel("Episode Reward")
    ax.set_title("NegotiateEnv — GRPO Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot_reward_curve] Saved → {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, help="Path to trainer_state.json or log list")
    parser.add_argument("--out", default="reward_curve.png")
    args = parser.parse_args()

    log_history = load_log_history(args.log_file)
    steps, rewards = extract_reward_series(log_history)

    if not rewards:
        print("[plot_reward_curve] No reward entries found in log. Check key names.", file=sys.stderr)
        sys.exit(1)

    print(f"[plot_reward_curve] Found {len(rewards)} reward entries. "
          f"Range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    plot(steps, rewards, args.out)


if __name__ == "__main__":
    main()
