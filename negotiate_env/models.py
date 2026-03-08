"""Pydantic models for NegotiateEnv: Action and Observation."""

from typing import Any

from openenv.core.env_server import Action, Observation


class NegotiateAction(Action):
    """Action from the agent (procurement manager) to the environment."""

    action_type: str  # "offer" | "counter" | "probe" | "accept" | "walkaway"
    price_per_seat: float = 0.0
    contract_length: float = 0.0
    annual_increase_cap: float = 0.0
    message: str = ""


class NegotiateObservation(Observation):
    """Observation returned to the agent after reset or step."""

    context: str = ""
    your_max_price: float = 0.0
    your_max_length: float = 0.0
    your_max_cap: float = 0.0
    ae_message: str = ""
    conversation_history: list[str] = []
    turn_number: int = 0
    max_turns: int = 0
    active_constraints: list[str] = []
    current_offer: dict[str, Any] = {}
    reward: float = 0.0
    done: bool = False
