"""FastAPI app for NegotiateEnv (OpenEnv v0.2.1) with WebSocket support."""

from openenv.core.env_server import create_fastapi_app, ConcurrencyConfig

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment

import os

_difficulty = os.environ.get("NEGOTIATE_DIFFICULTY", "medium")
_use_hf = os.environ.get("NEGOTIATE_USE_HF_DATASET", "true").lower() == "true"


def _env_factory() -> NegotiateEnvironment:
    return NegotiateEnvironment(difficulty=_difficulty, use_hf_dataset=_use_hf)


# Configure for WebSocket and concurrent HTTP sessions
concurrency_config = ConcurrencyConfig(
    max_concurrent_sessions=100,
    session_timeout_s=600,  # 10 minute timeout
    enable_session_cleanup=True,
)

app = create_fastapi_app(
    _env_factory, 
    NegotiateAction, 
    NegotiateObservation,
    concurrency_config=concurrency_config,
)

# WebSocket is automatically enabled by OpenEnv when SUPPORTS_CONCURRENT_SESSIONS = True
