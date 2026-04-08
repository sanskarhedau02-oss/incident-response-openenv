"""
AIOps OpenEnv — FastAPI Server
================================
Endpoints:
  GET  /                              — landing page (HTML)
  POST /reset?task=easy|medium|hard   — start a new episode
  POST /step                          — apply an action
  GET  /state                         — current raw state
  GET  /observation                   — normalised 12-float observation vector
  GET  /health                        — liveness check
  GET  /info                          — environment metadata
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.models import AIAction, StepResponse, ResetResponse, AIObservation
from server.environment import AIOpsEnvironment, MAX_STEPS, N_ACTIONS, OBS_DIM

app = FastAPI(
    title="AIOps OpenEnv",
    description="Incident-response simulation environment for RL agents.",
    version="2.0.0",
)

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

env: AIOpsEnvironment = AIOpsEnvironment(task="easy")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/info")
def info():
    return {
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "max_steps": MAX_STEPS,
        "tasks": ["easy", "medium", "hard"],
        "action_space": {
            "0": "noop",
            "1": "restart_service (auth-service)",
            "2": "restart_service (api)",
            "3": "scale_up",
            "4": "rollback",
            "5": "flush_cache",
        },
    }


@app.post("/reset", response_model=ResetResponse)
def reset(task: str = Query("easy", enum=["easy", "medium", "hard"])):
    global env
    env = AIOpsEnvironment(task=task)
    state = env.reset()
    return ResetResponse(observation=AIObservation(**state), reward=0.0, done=False, task=task)


@app.post("/step", response_model=StepResponse)
def step(action: AIAction):
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    state, reward, done, info = env.step(action.dict())
    return StepResponse(observation=AIObservation(**state), reward=reward, done=done, info=info)


@app.get("/state")
def state():
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return env.get_state()


@app.get("/observation")
def observation():
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return {"observation_vector": env.get_observation_vector(), "dim": OBS_DIM}


# Mount static files LAST so API routes take priority.
# html=True makes FastAPI serve index.html for "/" automatically.
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
