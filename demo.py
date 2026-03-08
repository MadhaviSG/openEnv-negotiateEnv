#!/usr/bin/env python3
"""Interactive demo for NegotiateEnv.

Shows a full negotiation transcript with reward breakdown.
Can run with a rule-based agent (default) or an LLM agent.

Usage:
    # Rule-based demo (no API key needed):
    python demo.py

    # Specific scenario:
    python demo.py --scenario slack_business_plus_200_seats

    # LLM demo:
    python demo.py --agent llm --model gpt-4o-mini

    # Hard difficulty:
    python demo.py --difficulty hard
"""

from __future__ import annotations

import argparse
import os
import random

from negotiate_env.client.negotiate_env_client import NegotiateEnvClient, observation_to_prompt
from negotiate_env.models import NegotiateAction, NegotiateObservation


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def rule_policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
    current_price = obs.current_offer.get("price_per_seat", 100.0)
    max_price = obs.your_max_price
    max_length = obs.your_max_length
    max_cap = obs.your_max_cap
    turns_left = obs.max_turns - turn

    if current_price <= max_price:
        return NegotiateAction(
            action_type="accept",
            price_per_seat=current_price,
            contract_length=obs.current_offer.get("contract_length", 2.0),
            annual_increase_cap=obs.current_offer.get("annual_increase_cap", 7.0),
            message="This works within our budget. Let's proceed.",
        )
    if turns_left <= 1:
        return NegotiateAction(action_type="walkaway",
                               message="We can't reach an agreement within our constraints.")
    if turn == 1:
        return NegotiateAction(action_type="probe",
                               message="What's the best you can do on price for a multi-year deal?")
    if turn == 2:
        return NegotiateAction(
            action_type="counter",
            price_per_seat=max_price,
            contract_length=min(2.0, max_length),
            annual_increase_cap=max_cap,
            message=f"Our budget is ${max_price:.0f}/seat on a 2-year with {max_cap:.0f}% cap.",
        )
    if turn == 3 and max_length >= 3.0:
        return NegotiateAction(
            action_type="counter",
            price_per_seat=max_price,
            contract_length=3.0,
            annual_increase_cap=max_cap,
            message=f"We'll commit to 3 years if you can meet ${max_price:.0f}/seat.",
        )
    target = max(max_price, current_price * 0.94)
    return NegotiateAction(
        action_type="counter",
        price_per_seat=round(target, 2),
        contract_length=min(2.0, max_length),
        annual_increase_cap=max_cap,
        message=f"We need to be at ${target:.0f}/seat to make this work.",
    )


def make_llm_policy(model: str):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    SYSTEM = (
        "You are a procurement manager negotiating a B2B SaaS contract. "
        "Respond with a JSON action: "
        '{"action_type": "counter", "price_per_seat": 85.0, '
        '"contract_length": 1.0, "annual_increase_cap": 3.0, "message": "..."}'
    )

    def policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
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
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(
    env: NegotiateEnvClient,
    policy,
    scenario_id: str | None,
    max_turns: int,
    difficulty: str,
) -> None:
    reset_kwargs: dict = {"max_turns": max_turns}
    if scenario_id:
        reset_kwargs["scenario_id"] = scenario_id

    obs = env.reset(**reset_kwargs)

    print("\n" + "=" * 65)
    print("  NegotiateEnv — Negotiation Demo")
    print(f"  Difficulty: {difficulty.upper()}")
    print("=" * 65)
    print(f"\nContext:\n  {obs.context}\n")
    print(f"Your constraints:")
    print(f"  Max price:  ${obs.your_max_price:.2f}/seat/month")
    print(f"  Max length: {obs.your_max_length:.0f} year(s)")
    print(f"  Max cap:    {obs.your_max_cap:.1f}%")
    print(f"\nVendor opening offer:")
    print(f"  Price:  ${obs.current_offer.get('price_per_seat', 0):.2f}/seat/month")
    print(f"  Length: {obs.current_offer.get('contract_length', 0):.0f} year(s)")
    print(f"  Cap:    {obs.current_offer.get('annual_increase_cap', 0):.1f}%")
    print(f"\nAE: \"{obs.ae_message}\"")
    print("-" * 65)

    turn = 0
    while not obs.done:
        turn += 1
        action = policy(obs, turn)

        # Print agent action
        if action.action_type in ("offer", "counter"):
            print(f"\n[Turn {turn}] Agent → {action.action_type.upper()}")
            print(f"  Offer: ${action.price_per_seat:.2f}/seat, "
                  f"{action.contract_length:.0f}y, {action.annual_increase_cap:.1f}% cap")
        else:
            print(f"\n[Turn {turn}] Agent → {action.action_type.upper()}")
        if action.message:
            print(f"  \"{action.message}\"")

        obs = env.step(action)

        if obs.active_constraints:
            print(f"\n  ⚠  CONSTRAINT DRIFT: {obs.active_constraints[-1]}")

        if not obs.done:
            print(f"\nAE: \"{obs.ae_message}\"")
            print(f"  Current offer: ${obs.current_offer.get('price_per_seat', 0):.2f}/seat, "
                  f"{obs.current_offer.get('contract_length', 0):.0f}y, "
                  f"{obs.current_offer.get('annual_increase_cap', 0):.1f}% cap")

    print("\n" + "=" * 65)
    print("  EPISODE COMPLETE")
    print("=" * 65)
    print(f"\nAE final: \"{obs.ae_message}\"")
    print(f"\nFinal reward: {obs.reward:.4f}")

    if obs.reward > 0:
        deal = obs.current_offer
        print(f"\nDeal struck:")
        print(f"  Price:  ${deal.get('price_per_seat', 0):.2f}/seat/month")
        print(f"  Length: {deal.get('contract_length', 0):.0f} year(s)")
        print(f"  Cap:    {deal.get('annual_increase_cap', 0):.1f}%")
        print(f"  Turns:  {turn}")
    else:
        print("\nNo deal reached.")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NegotiateEnv demo")
    parser.add_argument("--agent", choices=["rule", "llm"], default="rule")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--scenario", default=None, help="Scenario ID (optional)")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--env-url", default="http://127.0.0.1:7860")
    args = parser.parse_args()

    env = NegotiateEnvClient(base_url=args.env_url)

    if args.agent == "llm":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to use --agent llm")
        policy = make_llm_policy(args.model)
    else:
        policy = rule_policy

    run_demo(
        env=env,
        policy=policy,
        scenario_id=args.scenario,
        max_turns=args.max_turns,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
