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
from fastapi.responses import HTMLResponse, FileResponse
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


# Serve index.html explicitly at GET "/" only — avoids StaticFiles shadowing POST routes.
@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>AIOps OpenEnv</h1><p>Visit <a href='/docs'>/docs</a> for API.</p>")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

@app.get("/tasks")
def list_tasks():
    """List all tasks with grader info — required by the hackathon evaluator."""
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Single service down, low CPU, low error rate",
                "grader": True,
                "passing_threshold": 0.5
            },
            {
                "id": "medium",
                "description": "CPU spike, elevated error rate and latency, all services up",
                "grader": True,
                "passing_threshold": 0.4
            },
            {
                "id": "hard",
                "description": "Multi-service failure, cascading error rate, SEV-1 scenario",
                "grader": True,
                "passing_threshold": 0.3
            }
        ]
    }


@app.api_route("/grade", methods=["GET", "POST"])
def grade_task(task: str = "easy"):
    """Run a grader episode for the given task — required by the hackathon evaluator."""
    from server.environment import AIOpsEnvironment, MAX_STEPS

    thresholds = {"easy": 0.5, "medium": 0.4, "hard": 0.3}
    if task not in thresholds:
        return {"error": f"Unknown task: {task}. Choose from easy, medium, hard."}

    env = AIOpsEnvironment(task=task)
    obs = env.reset()

    total_reward = 0.0
    steps_taken = 0

    for _ in range(MAX_STEPS):
        # Heuristic agent for grading
        services = obs.get("services", {})
        cpu = obs.get("cpu_usage", 0)
        error_rate = obs.get("error_rate", 0)
        latency = obs.get("latency_ms", 0)

        action = {"action_type": "noop", "target": None}
        for svc in ["auth-service", "api", "cache", "db"]:
            if services.get(svc) not in ("healthy", None):
                action = {"action_type": "restart_service", "target": svc}
                break
        else:
            if cpu > 80:
                action = {"action_type": "scale_up", "target": None}
            elif latency > 800:
                action = {"action_type": "flush_cache", "target": None}
            elif error_rate > 0.25:
                action = {"action_type": "rollback", "target": None}

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps_taken += 1
        if done:
            break

    score = round(min(max(total_reward / max(steps_taken, 1), 0.0), 1.0), 4)

    return {
        "task": task,
        "score": score,
        "reward": round(total_reward, 4),
        "steps": steps_taken,
        "passed": score >= thresholds[task]
    }

if __name__ == "__main__":
    main()
