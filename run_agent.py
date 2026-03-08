"""LLM agent that plays through a NegotiateEnv negotiation using OpenAI."""

import os
import json
from openai import OpenAI
from negotiate_env.client.negotiate_env_client import (
    NegotiateEnvClient,
    observation_to_prompt,
    parse_llm_response_to_action,
)

BASE_URL = "http://127.0.0.1:8000"
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a skilled procurement manager negotiating a B2B SaaS contract on behalf of your company.
Your goal is to get the best possible deal within your budget constraints.

On each turn, read the observation carefully and respond with ONE action in this exact format:

Action: <offer|counter|probe|accept|walkaway>
Price: <$/seat/month, e.g. 195>
Length: <contract years: 1, 2, or 3>
Cap: <%annual increase cap, e.g. 4>
Message: <your message to the AE>

Guidelines:
- Use "probe" first to gather information if you're unsure of the AE's flexibility.
- Use "counter" to propose new terms.
- Use "accept" only when the current offer meets your constraints.
- Use "walkaway" only as a last resort if no deal is possible.
- Never exceed your max price, max length, or max cap constraints.
- Negotiate assertively but professionally.
"""


def parse_structured_response(text: str):
    """Parse structured action format from LLM."""
    import re

    action_type = "probe"
    price = 0.0
    length = 0.0
    cap = 0.0
    message = ""

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("offer", "counter", "probe", "accept", "walkaway"):
                action_type = val
        elif line.lower().startswith("price:"):
            nums = re.findall(r"[\d.]+", line)
            if nums:
                price = float(nums[0])
        elif line.lower().startswith("length:"):
            nums = re.findall(r"[\d.]+", line)
            if nums:
                length = float(nums[0])
        elif line.lower().startswith("cap:"):
            nums = re.findall(r"[\d.]+", line)
            if nums:
                cap = float(nums[0])
        elif line.lower().startswith("message:"):
            message = line.split(":", 1)[1].strip()

    from negotiate_env.models import NegotiateAction
    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=length,
        annual_increase_cap=cap,
        message=message,
    )


def run_agent(scenario_id: str = None, max_turns: int = 10):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY environment variable before running.")

    client = OpenAI(api_key=api_key)
    env = NegotiateEnvClient(base_url=BASE_URL)

    reset_kwargs = {"max_turns": max_turns}
    if scenario_id:
        reset_kwargs["scenario_id"] = scenario_id

    print("=" * 60)
    print("Starting negotiation episode...")
    print("=" * 60)

    obs = env.reset(**reset_kwargs)

    print(f"\nScenario context:\n{obs.context}\n")
    print(f"Your constraints: max price=${obs.your_max_price:.2f}, "
          f"max length={obs.your_max_length:.0f}y, max cap={obs.your_max_cap:.1f}%")
    print(f"\nAE opening: {obs.ae_message}\n")

    total_reward = 0.0

    while not obs.done:
        prompt = observation_to_prompt(obs)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        llm_text = response.choices[0].message.content
        print(f"--- Turn {obs.turn_number + 1} ---")
        print(f"LLM response:\n{llm_text}\n")

        action = parse_structured_response(llm_text)
        print(f"Parsed action: {action.action_type} | "
              f"price=${action.price_per_seat} length={action.contract_length}y "
              f"cap={action.annual_increase_cap}% | msg: {action.message[:80]}")

        obs = env.step(action)
        total_reward = obs.reward

        print(f"AE response: {obs.ae_message}")
        if obs.active_constraints:
            print(f"[!] New constraint: {obs.active_constraints[-1]}")
        print()

        if obs.done:
            break

    print("=" * 60)
    print("Episode complete.")
    print(f"Final reward: {total_reward:.4f}")
    if total_reward > 0:
        deal = obs.current_offer
        print(f"Deal struck: ${deal.get('price_per_seat', 0):.2f}/seat, "
              f"{deal.get('contract_length', 0):.0f}y, "
              f"{deal.get('annual_increase_cap', 0):.1f}% cap")
    else:
        print("No deal reached.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM negotiation agent")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario ID (optional)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max negotiation turns")
    args = parser.parse_args()
    run_agent(scenario_id=args.scenario, max_turns=args.max_turns)
