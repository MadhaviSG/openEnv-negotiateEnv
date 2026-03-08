"""FastAPI app for NegotiateEnv (OpenEnv v0.2.1)."""

from openenv.core.env_server import create_fastapi_app

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment

# create_fastapi_app expects a factory callable (Callable[[], Environment]).
# Passing the class itself satisfies this: NegotiateEnvironment() creates a fresh instance.
app = create_fastapi_app(NegotiateEnvironment, NegotiateAction, NegotiateObservation)
