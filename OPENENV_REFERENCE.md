# OpenEnv Framework Reference

> Official documentation extracted for NegotiateEnv implementation

## Main References

- **OpenEnv Core**: https://github.com/meta-pytorch/OpenEnv
- **Documentation**: https://meta-pytorch.org/OpenEnv/
- **HuggingFace**: https://huggingface.co/openenv
- **HF Spaces**: https://huggingface.co/openenv/spaces
- **Tutorials**: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial
- **Training Examples**: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples
- **Environment Examples**: https://github.com/meta-pytorch/OpenEnv/tree/main/envs

---

## OpenEnv Overview

OpenEnv is an end-to-end framework for creating, deploying, and training isolated execution environments for agentic reinforcement learning.

### Core Capabilities

1. Create environments with simple Python classes
2. Deploy environments anywhere (Docker, HuggingFace Spaces, cloud)
3. Train models with RL frameworks (TRL, Unsloth, TorchRL)
4. Run agents inside sandboxed containers

### Why OpenEnv Exists

**1. Fragmented Ecosystem**
- Problem: Agentic RL environments require multiple frameworks and tools
- Solution: Standardized interface integrating TRL, Unsloth, TorchRL, SkyRL

**2. Train vs Production Gap**
- Problem: Training pipelines differ from deployed agent systems
- Solution: Single environment interface for training and production

**3. No Isolation**
- Problem: Agents executing code can access host systems
- Solution: Each environment runs in isolated Docker container sandbox

**4. Scaling Difficulty**
- Problem: Running RL experiments requires specialized infrastructure
- Solution: Support for scalable environments using Docker, cloud workers, distributed simulations

---

## RL Basics

### Agent (Policy)
The learner that:
- Takes observations
- Outputs actions
- Often implemented as an LLM

### Environment
The world the agent interacts with. Returns:
- Observation
- Reward
- Done flag
- Metadata

### Action
What the agent decides to do. Examples:
- Tool call
- Code execution
- Text response

### Observation
Information returned from environment after action. Includes:
- State
- Reward
- Termination signal

### Episode
Multiple steps: `reset → step → step → step → done`
Goal: Maximize cumulative reward

---

## RL Training Loop in OpenEnv

```python
import asyncio
from hack_env import HackEnv

async def train(session_env, model):
    obs = await session_env.reset()
    trajectory = []
    
    while not obs.done:
        action = model.predict(obs)
        result = await env.step(action)
        trajectory.append({
            "obs": obs,
            "action": action,
            "reward": result.reward
        })
        obs = result.observation
    
    loss = rl_loss(trajectory)
    loss.backward()
```

Steps:
1. Reset environment
2. Agent chooses action
3. Environment executes action
4. Collect trajectory
5. Compute RL loss
6. Update model

---

## OpenEnv Architecture

```
Agent LLM
    ↓
Training Loop
    ↓
Gym-like API
    ↓
Environment Runtime
    ↓
Docker Container Sandbox
```

### Two APIs

**1. Training API**
- Used by RL trainers (TRL, TorchRL, Unsloth)

**2. Agent API**
- Used by agent clients (LLMs)
- Communication via WebSocket / JSON-RPC

---

## Creating an Environment

### File Structure

```
env/
├── client.py
├── server.py
├── models.py
├── requirements.txt
└── Dockerfile
```

### Example Models

```python
from openenv import Action, Observation

class EchoAction(Action):
    message: str

class EchoObservation(Observation):
    output: str
```

### Environment Implementation

```python
class EchoEnv:
    async def reset(self):
        return {"message": ""}
    
    async def step(self, action):
        return {
            "output": action.message
        }
```

### Client Environment

```python
from openenv import Client

async with Client(base_url="https://env.hf.space") as client:
    result = await client.reset()
    result = await client.step(action)
```

---

## Deploying Environments

### Local Docker

```bash
docker build -t openenv-env .
docker run -p 8000:8000 openenv-env
```

### HuggingFace Spaces

Push environment to HF. Each Space provides:
1. **Server endpoint**
2. **Python package repository**
3. **Docker container registry**

#### 1. Server

```bash
curl https://openenv-echo-env.hf.space/health
# {"status": "healthy"}
```

#### 2. Repository

```bash
pip install git+https://huggingface.co/spaces/openenv/echo-env
```

Installs:
- Client class
- Typed action models
- Observation models

#### 3. Docker Registry

```bash
docker pull registry.hf.space/openenv-echo-env:latest
docker run -p 8001:8000 registry.hf.space/openenv-echo-env:latest
```

---

## Development Workflow

### Development Mode
```python
client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
```

### Production Mode
```python
client = EchoEnv(base_url="http://localhost:8001")
```

### Auto-Docker Mode
```python
client = EchoEnv.from_env("openenv/echo-env")
```

---

## Scaling Environments

### Concurrent Sessions
Multiple sessions in one environment.
```python
session_pool = 2000
```

### Docker Swarm Replicas
Horizontal scaling.
```bash
docker service scale openenv=10
```

### Cloud Worker Pools
Batch parallel training using:
- Ray
- Kubernetes
- Slurm

---

## Loading Environments

### Local Server
```python
env = EchoEnv(base_url="http://localhost:8001")
```

### Docker Image
```python
env = EchoEnv.from_docker("openenv/echo-env")
```

### HuggingFace Hub
```python
env = EchoEnv.from_env("openenv/echo-env")
```

### Local Provider
```python
provider = LocalProvider()
env = provider.start()
```

---

## Training with TRL

TRL supports OpenEnv environments.

**Documentation**: https://huggingface.co/docs/trl/en/openenv

**Examples**:
- Sudoku RL training
- Wordle RL training

---

## Training with Unsloth

**Example notebook**: https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/examples/unsloth_2048.ipynb

Supports:
- GRPO
- PPO
- RLHF training

---

## Hackathon Infrastructure

### GPU Infrastructure
- **120 × NVIDIA H100 GPUs**
- **1 GPU per team**
- Provided by: CoreWeave + Northflank

### GPU Access Setup

Teams must complete GPU request form:
https://docs.google.com/forms/d/e/1FAIpQLSd2bxx5jAXE8D3FjF7OVekSxwpDVMf1LWE3Z-g4FZoDJ4W6xg/viewform

### Northflank Deployment Options

Teams receive a project workspace. Possible deployments:
- Jupyter Notebook with PyTorch
- Ubuntu container shell
- Docker deployment from GitHub
- Public AI Docker images

### SSH Access

To access GPU node:
1. Install Node.js + Northflank CLI
2. Connect via SSH

### VSCode Remote Development

Connect to container using:
- VSCode Remote SSH
- Cursor IDE

Use: `ip`, `port`, `root` user from Northflank terminal.

### Jupyter Notebook Option

Northflank supports one-click Jupyter deployment.

Useful for:
- Unsloth training
- RL experimentation

---

## Support

Questions can be asked on the hackathon Discord.

---

## NegotiateEnv Alignment Checklist

- ✅ Gymnasium-style API (`reset()`, `step()`, `state()`)
- ✅ Pydantic models for Action and Observation
- ✅ FastAPI server via `create_fastapi_app`
- ✅ WebSocket client
- ✅ Docker deployment (multi-stage, uv-based)
- ✅ HuggingFace Spaces compatible (port 7860)
- ✅ TRL GRPO training script
- ✅ Unsloth training script
- ✅ Concurrent sessions support (`SUPPORTS_CONCURRENT_SESSIONS = True`)
- ✅ Isolated execution (server-side hidden state)
- ✅ Scalable (stateless environment, multiple parallel rollouts)
