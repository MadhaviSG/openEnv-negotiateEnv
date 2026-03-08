"""FastAPI app for NegotiateEnv (OpenEnv v0.2.1)."""

from openenv_core import create_fastapi_app

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment

app = create_fastapi_app(NegotiateEnvironment, NegotiateAction, NegotiateObservation)
