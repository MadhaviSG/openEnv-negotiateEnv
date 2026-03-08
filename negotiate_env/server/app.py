"""FastAPI app for NegotiateEnv (OpenEnv v0.2.1)."""

from openenv.core.env_server import create_fastapi_app

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment

# create_fastapi_app expects a factory callable (Callable[[], Environment]).
# Passing the class itself satisfies this: NegotiateEnvironment() creates a fresh instance.
# Default difficulty is "medium". To change, set NEGOTIATE_DIFFICULTY env var.
import os

_difficulty = os.environ.get("NEGOTIATE_DIFFICULTY", "medium")
_use_hf = os.environ.get("NEGOTIATE_USE_HF_DATASET", "false").lower() == "true"


def _env_factory() -> NegotiateEnvironment:
    return NegotiateEnvironment(difficulty=_difficulty, use_hf_dataset=_use_hf)


app = create_fastapi_app(_env_factory, NegotiateAction, NegotiateObservation)
