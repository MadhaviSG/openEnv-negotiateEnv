"""Fixed FastAPI app with proper session management."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uuid

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment

app = FastAPI(title="NegotiateEnv API")

# In-memory session storage
sessions: Dict[str, NegotiateEnvironment] = {}

class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action: NegotiateAction
    session_id: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset(request: ResetRequest = None):
    """Reset environment and return session_id."""
    if request and request.session_id:
        session_id = request.session_id
    else:
        session_id = str(uuid.uuid4())
    
    env = NegotiateEnvironment(difficulty="medium", use_hf_dataset=True)
    kwargs = {}
    if request and request.scenario_id:
        kwargs["scenario_id"] = request.scenario_id
    
    obs = env.reset(**kwargs)
    sessions[session_id] = env
    
    return {
        "session_id": session_id,
        "observation": obs.dict(),
        "reward": obs.reward,
        "done": obs.done,
    }

@app.post("/step")
def step(request: StepRequest):
    """Take action in environment."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id. Call /reset first.")
    
    env = sessions[request.session_id]
    obs = env.step(request.action)
    
    # Clean up finished sessions
    if obs.done:
        del sessions[request.session_id]
    
    return {
        "observation": obs.dict(),
        "reward": obs.reward,
        "done": obs.done,
    }

@app.get("/state")
def state():
    """Get server state."""
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys())[:5],  # Show first 5
    }
