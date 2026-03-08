"""Synchronous WebSocket client for NegotiateEnv server.

Connects via persistent WebSocket to the NegotiateEnv FastAPI server,
formats observations as LLM prompts, and parses LLM text output into
NegotiateAction objects.
"""

import json
import re
from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from negotiate_env.models import NegotiateAction, NegotiateObservation


class NegotiateEnvClient(EnvClient[NegotiateAction, NegotiateObservation, Dict[str, Any]]):
    """Synchronous WebSocket client for the NegotiateEnv server.

    Maintains a persistent WebSocket connection so state is shared across
    reset() and step() calls (required for stateful multi-turn negotiation).

    Example:
        env = NegotiateEnvClient("http://localhost:7860")
        obs = env.reset()
        while not obs.done:
            action = env.parse_llm_output_to_action(llm_response)
            obs = env.step(action)
        env.close()
    """

    # ------------------------------------------------------------------
    # EnvClient abstract interface
    # ------------------------------------------------------------------

    def _step_payload(self, action: NegotiateAction) -> Dict[str, Any]:
        """Serialize NegotiateAction to the dict the server expects."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[NegotiateObservation]:
        """
        Parse server WebSocket response into a typed StepResult.

        serialize_observation() puts fields into:
          {"observation": {all fields except reward/done/metadata},
           "reward": float|None,
           "done": bool}

        We merge reward and done back into the obs dict so NegotiateObservation
        validates cleanly (it inherits those fields from Observation).
        """
        obs_data = payload.get("observation", {})
        reward = payload.get("reward") or 0.0
        done = bool(payload.get("done", False))
        full_obs_data = {**obs_data, "reward": reward, "done": done}
        obs = NegotiateObservation.model_validate(full_obs_data)
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    # ------------------------------------------------------------------
    # Public API — returns NegotiateObservation directly (not StepResult)
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> NegotiateObservation:
        """Reset the environment and return the initial observation.

        Auto-connects the WebSocket on first call if not already connected.
        """
        if self._ws is None:
            self.connect()
        result = super().reset(**kwargs)
        return result.observation

    def step(self, action: NegotiateAction, **kwargs) -> NegotiateObservation:
        """Send an action and return the resulting observation."""
        result = super().step(action, **kwargs)
        return result.observation

    # ------------------------------------------------------------------
    # Prompt formatting and LLM output parsing (static — no server needed)
    # ------------------------------------------------------------------

    @staticmethod
    def format_observation_as_prompt(obs: NegotiateObservation) -> str:
        """Format observation into a clean LLM-readable negotiation state string."""
        turns_remaining = obs.max_turns - obs.turn_number
        current = obs.current_offer

        lines = [
            "## Negotiation Context",
            obs.context,
            "",
            "## AE's Current Offer",
            f"  Price:    ${current.get('price_per_seat', 0):.2f}/seat/month",
            f"  Length:   {current.get('contract_length', 0):.0f} year(s)",
            f"  Ann. cap: {current.get('annual_increase_cap', 0):.1f}%",
            "",
            "## Your Budget Limits",
            f"  Max price:   ${obs.your_max_price:.2f}/seat/month",
            f"  Max length:  {obs.your_max_length:.0f} year(s)",
            f"  Max cap:     {obs.your_max_cap:.1f}%",
        ]

        if obs.active_constraints:
            lines += ["", "## Active Constraints (NEW)"]
            for c in obs.active_constraints:
                lines.append(f"  ⚠ {c}")

        history_tail = obs.conversation_history[-6:] if obs.conversation_history else []
        if history_tail:
            lines += ["", "## Recent Conversation"]
            lines += [f"  {h}" for h in history_tail]

        lines += [
            "",
            f"## Turn {obs.turn_number} of {obs.max_turns} ({turns_remaining} remaining)",
            "",
            "AE just said:",
            f'  "{obs.ae_message}"',
            "",
            "Respond with a JSON action:",
            '  {"action_type": "counter", "price_per_seat": 85.0, '
            '"contract_length": 1.0, "annual_increase_cap": 3.0, "message": "..."}',
            "action_type must be one of: offer | counter | probe | accept | walkaway",
        ]
        return "\n".join(lines)

    @staticmethod
    def parse_llm_output_to_action(text: str) -> NegotiateAction:
        """Parse LLM text output into a NegotiateAction.

        Tries JSON first, then regex extraction, then falls back to a safe default.
        """
        text = text.strip()

        # 1) Try full JSON parse
        try:
            data = json.loads(text)
            return NegotiateAction(**{k: v for k, v in data.items() if k in NegotiateAction.model_fields})
        except Exception:
            pass

        # 2) Try to extract JSON object with regex
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return NegotiateAction(**{k: v for k, v in data.items() if k in NegotiateAction.model_fields})
            except Exception:
                pass

        # 3) Heuristic extraction
        action_type = "counter"
        for at in ("accept", "walkaway", "probe", "offer", "counter"):
            if at in text.lower():
                action_type = at
                break

        price = 0.0
        length = 1.0
        cap = 5.0

        dollar = re.search(r"\$\s*([\d]+(?:\.\d+)?)", text)
        if dollar:
            price = float(dollar.group(1))

        nums = re.findall(r"[\d]+(?:\.\d+)?", text)
        if not dollar and len(nums) >= 1 and action_type in ("offer", "counter"):
            price = float(nums[0])
        if len(nums) >= 2:
            candidate = float(nums[1])
            if 1.0 <= candidate <= 5.0:
                length = candidate
        if len(nums) >= 3:
            candidate = float(nums[2])
            if 0.0 <= candidate <= 15.0:
                cap = candidate

        return NegotiateAction(
            action_type=action_type,
            price_per_seat=price,
            contract_length=length,
            annual_increase_cap=cap,
            message=text[:200],
        )

    # Back-compat alias
    def parse_llm_response_to_action(self, text: str) -> NegotiateAction:
        return self.parse_llm_output_to_action(text)


# ---------------------------------------------------------------------------
# Module-level helpers imported by run_agent.py and train scripts
# ---------------------------------------------------------------------------

def observation_to_prompt(obs: NegotiateObservation) -> str:
    """Format a NegotiateObservation as an LLM-readable prompt (module-level)."""
    return NegotiateEnvClient.format_observation_as_prompt(obs)


def parse_llm_response_to_action(text: str) -> NegotiateAction:
    """Parse LLM text output into a NegotiateAction (module-level alias)."""
    return NegotiateEnvClient.parse_llm_output_to_action(text)
