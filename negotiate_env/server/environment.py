"""NegotiateEnvironment: OpenEnv Environment for B2B SaaS contract negotiation."""

import random
from typing import Any, Optional

from openenv.core.env_server import Environment

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.scenarios import SCENARIOS
from negotiate_env.server.opponent import AEOpponent


class NegotiateEnvironment(Environment):
    """RL environment: agent (procurement manager) negotiates with rule-based AE."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._step_count: int = 0
        self._scenario: dict[str, Any] = {}
        self._opponent: Optional[AEOpponent] = None
        self._vendor_floor_price: float = 0.0
        self._vendor_list_price: float = 0.0
        self._vendor_preferred_length: float = 0.0
        self._vendor_max_cap: float = 10.0
        self._vendor_min_cap: float = 3.0
        self._agent_max_price: float = 0.0
        self._agent_max_length: float = 0.0
        self._agent_max_cap: float = 0.0
        self._max_turns: int = 10
        self._enable_drift: bool = True
        self._conversation_history: list[str] = []
        self._current_offer: dict[str, Any] = {}
        self._active_constraints: list[str] = []
        self._drift_injected: bool = False
        self._turn_penalties: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> NegotiateObservation:
        config = kwargs
        if seed is not None:
            random.seed(seed)
        scenario_id = config.get("scenario_id")
        if scenario_id:
            self._scenario = next(
                (s for s in SCENARIOS if s["id"] == scenario_id),
                random.choice(SCENARIOS),
            )
        else:
            self._scenario = random.choice(SCENARIOS)

        self._max_turns = int(config.get("max_turns", 10))
        self._enable_drift = config.get("enable_drift", True)

        # Hidden internal state — NEVER exposed in observations
        self._vendor_floor_price = self._scenario["vendor_floor_price"]
        self._vendor_list_price = self._scenario["vendor_list_price"]
        self._vendor_preferred_length = self._scenario["vendor_preferred_length"]
        self._vendor_max_cap = self._scenario.get("vendor_max_cap", 10.0)
        self._vendor_min_cap = self._scenario.get("vendor_min_cap", 3.0)

        self._agent_max_price = self._scenario["agent_max_price"]
        self._agent_max_length = self._scenario["agent_max_length"]
        self._agent_max_cap = self._scenario["agent_max_cap"]

        self._opponent = AEOpponent(self._scenario, self._scenario["opponent_strategy"])
        self._current_offer = {
            "price_per_seat": self._vendor_list_price,
            "contract_length": self._vendor_preferred_length,
            "annual_increase_cap": self._vendor_max_cap,
        }
        self._conversation_history = []
        self._active_constraints = []
        self._drift_injected = False
        self._turn_penalties = 0.0
        self._step_count = 0

        opening_msg = self._scenario["vendor_opening_message"]
        self._conversation_history.append(f"AE: {opening_msg}")

        return NegotiateObservation(
            context=self._scenario["context"],
            your_max_price=self._agent_max_price,
            your_max_length=self._agent_max_length,
            your_max_cap=self._agent_max_cap,
            ae_message=opening_msg,
            conversation_history=list(self._conversation_history),
            turn_number=0,
            max_turns=self._max_turns,
            active_constraints=[],
            current_offer=dict(self._current_offer),
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: NegotiateAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> NegotiateObservation:
        self._step_count += 1
        turn = self._step_count
        self._turn_penalties += 0.01

        # Inject drift exactly once at drift_turn
        if (
            self._enable_drift
            and not self._drift_injected
            and turn >= self._scenario.get("drift_turn", 999)
        ):
            self._active_constraints.append(self._scenario["drift_event"])
            self._drift_injected = True

        # Record agent's move in history
        if action.action_type in ("offer", "counter") and (
            action.price_per_seat or action.contract_length or action.annual_increase_cap
        ):
            self._conversation_history.append(
                f"Agent: [Offer] ${action.price_per_seat:.0f}/seat, "
                f"{action.contract_length:.0f}y, {action.annual_increase_cap:.1f}% cap"
            )
        elif action.message:
            self._conversation_history.append(f"Agent: {action.message}")

        if action.action_type == "walkaway":
            return self._obs(
                ae_message="I'm sorry we couldn't find common ground. Feel free to reach out if things change.",
                done=True,
                reward=0.0,
            )

        if action.action_type == "accept":
            if self._current_offer.get("price_per_seat", 0) >= self._vendor_floor_price:
                reward = self._compute_reward(self._current_offer)
                return self._obs(
                    ae_message="Perfect. We're aligned. I'll get the contract over to you.",
                    done=True,
                    reward=reward,
                )
            # Invalid accept — AE counter-offers because their standing offer is below floor threshold
            ae_msg, self._current_offer = self._opponent.respond(
                action, turn, self._conversation_history, self._current_offer
            )
            self._conversation_history.append(f"AE: {ae_msg}")
            return self._obs(ae_message=ae_msg, done=False, reward=0.0)

        if action.action_type == "probe":
            ae_msg, _ = self._opponent._probe_response(self._current_offer)
            self._conversation_history.append(f"AE: {ae_msg}")
            return self._obs(ae_message=ae_msg, done=False, reward=0.0)

        # "offer" or "counter" — check if agent's numbers clear the floor
        agent_price = action.price_per_seat if action.price_per_seat > 0 else self._current_offer.get("price_per_seat", 0)
        agent_length = action.contract_length if action.contract_length > 0 else self._current_offer.get("contract_length", 2.0)
        agent_cap = action.annual_increase_cap if action.annual_increase_cap > 0 else self._current_offer.get("annual_increase_cap", 7.0)

        # AE accepts if: price >= floor, length >= 1yr, cap >= AE's minimum cap
        if (
            agent_price >= self._vendor_floor_price
            and agent_length >= 1.0
            and agent_cap >= self._vendor_min_cap
        ):
            deal = {
                "price_per_seat": agent_price,
                "contract_length": max(1.0, min(3.0, agent_length)),
                "annual_increase_cap": agent_cap,
            }
            self._current_offer = deal
            reward = self._compute_reward(deal)
            return self._obs(
                ae_message="We have a deal. I'll send over the paperwork at those terms.",
                done=True,
                reward=reward,
            )

        # Offer doesn't clear floor — AE counter-offers
        ae_msg, new_offer = self._opponent.respond(
            action, turn, self._conversation_history, self._current_offer
        )
        self._current_offer = new_offer
        self._conversation_history.append(f"AE: {ae_msg}")

        # Safety net: opponent may also accept in edge cases
        if any(kw in ae_msg.lower() for kw in ("we have a deal", "paperwork", "we're aligned")):
            reward = self._compute_reward(self._current_offer)
            return self._obs(ae_message=ae_msg, done=True, reward=reward)

        if turn >= self._max_turns:
            return self._obs(ae_message=ae_msg, done=True, reward=0.0)

        return self._obs(ae_message=ae_msg, done=False, reward=0.0)

    def _obs(
        self,
        ae_message: str,
        done: bool,
        reward: float,
    ) -> NegotiateObservation:
        return NegotiateObservation(
            context=self._scenario["context"],
            your_max_price=self._agent_max_price,
            your_max_length=self._agent_max_length,
            your_max_cap=self._agent_max_cap,
            ae_message=ae_message,
            conversation_history=list(self._conversation_history),
            turn_number=self._step_count,
            max_turns=self._max_turns,
            active_constraints=list(self._active_constraints),
            current_offer=dict(self._current_offer),
            reward=reward,
            done=done,
        )

    def _compute_reward(self, deal: dict[str, Any]) -> float:
        vendor_list = self._vendor_list_price
        vendor_floor = self._vendor_floor_price
        agent_target_cap = self._agent_max_cap
        vendor_max_cap = self._vendor_max_cap

        deal_price = deal.get("price_per_seat", vendor_list)
        deal_length = max(1.0, min(3.0, deal.get("contract_length", 2.0)))
        deal_cap = deal.get("annual_increase_cap", vendor_max_cap)

        price_range = vendor_list - vendor_floor
        price_score = (vendor_list - deal_price) / price_range if price_range > 0 else 1.0
        price_score = max(0.0, min(1.0, price_score))

        length_score = 1.0 - (deal_length - 1.0) / 2.0
        length_score = max(0.0, min(1.0, length_score))

        cap_range = vendor_max_cap - agent_target_cap
        cap_score = (vendor_max_cap - deal_cap) / cap_range if cap_range > 0 else 1.0
        cap_score = max(0.0, min(1.0, cap_score))

        raw = 0.5 * price_score + 0.3 * length_score + 0.2 * cap_score
        return round(max(0.0, raw - self._turn_penalties), 4)

    @property
    def state(self) -> dict[str, Any]:
        """Minimal state dict; openenv base class marks this abstract."""
        return {"step_count": self._step_count}
