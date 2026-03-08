"""Rule-based AE (Account Executive) opponent with hidden reservation price."""

import random
from typing import Any

from negotiate_env.models import NegotiateAction


class AEOpponent:
    """Account Executive opponent. Never accepts below vendor_floor_price."""

    def __init__(self, scenario: dict, strategy: str):
        self.scenario = scenario
        self.strategy = strategy
        self._floor_price = scenario["vendor_floor_price"]
        self._list_price = scenario["vendor_list_price"]
        self._preferred_length = scenario["vendor_preferred_length"]
        self._max_cap = scenario.get("vendor_max_cap", 10.0)   # AE's opening cap
        self._min_cap = scenario.get("vendor_min_cap", 3.0)    # AE's minimum acceptable cap

    def respond(
        self,
        action: NegotiateAction,
        turn: int,
        conversation_history: list[str],
        current_offer: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Returns (ae_message: str, current_offer: dict).
        current_offer has keys: price_per_seat, contract_length, annual_increase_cap.
        """
        offer = current_offer or self._get_default_offer()

        # Check if agent's offer (on offer/counter) meets or exceeds floor on all dimensions
        if action.action_type in ("offer", "counter"):
            agent_price = action.price_per_seat if action.price_per_seat > 0 else offer.get("price_per_seat", self._list_price)
            agent_length = action.contract_length if action.contract_length > 0 else offer.get("contract_length", self._preferred_length)
            agent_cap = action.annual_increase_cap if action.annual_increase_cap > 0 else offer.get("annual_increase_cap", self._max_cap)
            # Accept if price >= floor AND cap >= AE's minimum (AE is seller who wants high cap)
            if agent_price >= self._floor_price and agent_length >= 1.0 and agent_cap >= self._min_cap:
                return (
                    "We have a deal. I'll send over the paperwork at those terms.",
                    {
                        "price_per_seat": agent_price,
                        "contract_length": max(1.0, min(3.0, agent_length)),
                        "annual_increase_cap": agent_cap,
                    },
                )

        if action.action_type == "probe":
            return self._probe_response(offer)
        if action.action_type == "walkaway":
            return (
                "I'm sorry we couldn't find common ground. Feel free to reach out if things change.",
                offer,
            )
        if action.action_type == "accept":
            # Agent accepts our standing offer - only accept if offer is at or above floor
            if offer.get("price_per_seat", 0) >= self._floor_price:
                return (
                    "Perfect. We're aligned. I'll get the contract over to you.",
                    offer,
                )
            # Otherwise counter again
            return self._strategy_response(action, turn, offer, conversation_history)

        # offer or counter that didn't clear floor
        return self._strategy_response(action, turn, offer, conversation_history)

    def _get_default_offer(self) -> dict[str, Any]:
        """Default standing offer (list price)."""
        return {
            "price_per_seat": self._list_price,
            "contract_length": self._preferred_length,
            "annual_increase_cap": 7.0,
        }

    def _probe_response(self, offer: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Vague partial info — hints above the real floor, never reveals it."""
        # Hint is floor + 10% of the floor-to-list range (above the real floor, not at it)
        hint_price = self._floor_price + (self._list_price - self._floor_price) * 0.1
        return (
            f"I can't go below ${hint_price:.0f} on the per-seat without manager approval. "
            "If you can move on term length or the annual cap, we might get closer.",
            offer,
        )

    def _strategy_response(
        self,
        action: NegotiateAction,
        turn: int,
        current_offer: dict[str, Any],
        conversation_history: list[str],
    ) -> tuple[str, dict[str, Any]]:
        if self.strategy == "hardball":
            return self._hardball(action, turn, current_offer)
        if self.strategy == "concession_trader":
            return self._concession_trader(action, turn, current_offer)
        if self.strategy == "urgency":
            return self._urgency(action, turn, current_offer)
        return self._cooperative(action, turn, current_offer)

    def _hardball(
        self,
        action: NegotiateAction,
        turn: int,
        offer: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # Concedes very slowly (max 3–5% per turn), "final offer" at turn 4 even when it isn't
        price = offer.get("price_per_seat", self._list_price)
        length = offer.get("contract_length", self._preferred_length)
        cap = offer.get("annual_increase_cap", 7.0)
        concession = (self._list_price - self._floor_price) * 0.04  # ~4% of range per turn
        new_price = max(self._floor_price, price - concession)
        if turn >= 4:
            msg = "This is my final offer. I've gone as low as I can—$%.0f per seat, %s years, %s%% cap. Take it or we'll have to pause." % (
                new_price, int(length), int(cap)
            )
        else:
            msg = "We're already stretched. I can maybe do $%.0f per seat on a %s-year with %s%% cap. Beyond that I'd need to go back to my manager." % (
                new_price, int(length), int(cap)
            )
        new_offer = {"price_per_seat": new_price, "contract_length": length, "annual_increase_cap": cap}
        return (msg, new_offer)

    def _concession_trader(
        self,
        action: NegotiateAction,
        turn: int,
        offer: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # Drops price only if agent extends length or accepts higher cap
        price = offer.get("price_per_seat", self._list_price)
        length = offer.get("contract_length", self._preferred_length)
        cap = offer.get("annual_increase_cap", 7.0)
        agent_len = action.contract_length if action.contract_length > 0 else length
        agent_cap = action.annual_increase_cap if action.annual_increase_cap > 0 else cap
        new_price = price
        new_length = length
        new_cap = cap
        if agent_len >= 2.5 and length < 3.0:
            new_length = 3.0
            new_price = max(self._floor_price, price - (self._list_price - self._floor_price) * 0.15)
        if agent_cap >= 6.0 and cap < 7.0:
            new_cap = 7.0
            new_price = max(self._floor_price, price - (self._list_price - self._floor_price) * 0.08)
        if new_price == price and new_length == length:
            new_price = max(self._floor_price, price - (self._list_price - self._floor_price) * 0.05)
        msg = "If you can do a three-year or lock in a 7%% cap, I can move to $%.0f per seat. Otherwise we're stuck at list." % (new_price,)
        return (msg, {"price_per_seat": new_price, "contract_length": new_length, "annual_increase_cap": new_cap})

    def _urgency(
        self,
        action: NegotiateAction,
        turn: int,
        offer: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # Time pressure; less willing to move on price
        price = offer.get("price_per_seat", self._list_price)
        length = offer.get("contract_length", self._preferred_length)
        cap = offer.get("annual_increase_cap", 7.0)
        small_concession = (self._list_price - self._floor_price) * 0.03
        new_price = max(self._floor_price, price - small_concession)
        phrases = [
            "The quarter ends Friday and we have one slot left at this rate.",
            "We've got limited availability at this pricing for the rest of the year.",
            "If you can decide this week I might be able to hold this rate.",
        ]
        msg = random.choice(phrases) + " I can do $%.0f per seat on a %s-year, %s%% cap." % (new_price, int(length), int(cap))
        return (msg, {"price_per_seat": new_price, "contract_length": length, "annual_increase_cap": cap})

    def _cooperative(
        self,
        action: NegotiateAction,
        turn: int,
        offer: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # Find middle ground; respond to agent's constraints
        price = offer.get("price_per_seat", self._list_price)
        length = offer.get("contract_length", self._preferred_length)
        cap = offer.get("annual_increase_cap", 7.0)
        mid = (price + self._floor_price) / 2
        new_price = max(self._floor_price, min(price - (self._list_price - self._floor_price) * 0.12, mid))
        if action.message and ("length" in action.message.lower() or "year" in action.message.lower()):
            new_length = max(1.0, length - 0.5) if length > 1.5 else length
        else:
            new_length = length
        if action.message and ("cap" in action.message.lower() or "increase" in action.message.lower()):
            new_cap = min(10.0, cap - 0.5) if cap > 5.0 else cap
        else:
            new_cap = cap
        msg = "I hear you. I can move to $%.0f per seat, %s years, %s%% cap. Does that get us closer?" % (new_price, int(new_length), int(new_cap))
        return (msg, {"price_per_seat": new_price, "contract_length": new_length, "annual_increase_cap": new_cap})
