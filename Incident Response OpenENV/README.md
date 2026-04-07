# 🤖 AI Ops OpenEnv — PyTorch Hackathon Submission

> **Scaler School of Technology | PyTorch Hackathon**  
> Autonomous incident-response agent trained with Deep Q-Networks

---

## Overview

**AI Ops OpenEnv** is a production-grade simulation environment where a PyTorch-powered RL agent learns to autonomously remediate cloud infrastructure incidents — restarting failed services, scaling compute, rolling back bad deploys, and flushing caches — all without human intervention.

The environment follows the **OpenEnv** specification and is compatible with any agent that speaks HTTP or the Python client.

---

## Key Features

| Feature | Detail |
|---|---|
| **DQN Agent** | Double DQN + Dueling architecture (Mnih 2015, Wang 2016) |
| **Curriculum Learning** | Trains easy → medium → hard, promoting on 75% success rate |
| **Rich Environment** | 12-dim observation, 6 actions, stochastic noise, cascading failures |
| **Reward Shaping** | Per-step health bonuses + shaped action rewards |
| **REST API** | FastAPI server, full OpenAPI docs at `/docs` |
| **Heuristic Baseline** | Priority rule agent for comparison |
| **Checkpointing** | Best model saved automatically during training |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        DQN Agent                        │
│  ┌──────────────┐    ┌─────────────────────────────┐   │
│  │  Replay      │    │        DuelingDQN            │   │
│  │  Buffer      │    │  Encoder (MLP)               │   │
│  │  (15k steps) │    │   ├─ Value Stream  V(s)      │   │
│  └──────────────┘    │   └─ Advantage A(s,a)        │   │
│                       │  Q = V + A − mean(A)         │   │
│                       └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
           │  12-float observation vector
           ▼
┌─────────────────────────────────────────────────────────┐
│                  AIOps Environment                       │
│  Services: auth-service | api | db | cache               │
│  Metrics:  cpu_usage | memory | error_rate | latency     │
│  Tasks:    easy → medium → hard                          │
└─────────────────────────────────────────────────────────┘
```

---

## Observation Space (12 floats, all ∈ [0,1])

| Idx | Feature | Description |
|-----|---------|-------------|
| 0 | `cpu_usage` | CPU utilisation / 100 |
| 1 | `memory_usage` | Memory utilisation / 100 |
| 2 | `error_rate` | Fraction of failing requests |
| 3 | `latency_ms` | p99 latency / 2000 |
| 4–7 | `svc_*_healthy` | Binary health for each service |
| 8 | `step_progress` | step_count / MAX_STEPS |
| 9 | `severity` | incident_severity / 3 |
| 10 | `alerts` | alert_count / 10 |
| 11 | `restart_budget` | Remaining restart tokens / 5 |

## Action Space (6 discrete)

| Idx | Action | Effect |
|-----|--------|--------|
| 0 | `noop` | No-op (small negative reward) |
| 1 | `restart_service` → auth-service | Restores auth-service |
| 2 | `restart_service` → api | Restores api |
| 3 | `scale_up` | Reduces CPU & latency |
| 4 | `rollback` | Reduces error_rate |
| 5 | `flush_cache` | Reduces latency, minor error spike |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set mandatory environment variables

Per hackathon spec, these **must** be set before running inference:

```bash
export API_BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
export MODEL_NAME="gpt-4o-mini"                    # or e.g. meta-llama/Llama-3-8b-instruct
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"          # your Hugging Face / API key
```

### 3. Run inference (LLM agent — default)

```bash
# Run LLM agent on easy task (uses API_BASE_URL / MODEL_NAME / HF_TOKEN)
python inference.py --task easy --mode llm --episodes 3

# Run heuristic baseline (no API key needed)
python inference.py --task easy --mode heuristic --episodes 5

# Run all three tasks
python inference.py --task easy   --mode llm --episodes 3
python inference.py --task medium --mode llm --episodes 3
python inference.py --task hard   --mode llm --episodes 3
```

### 4. Train the DQN agent

```bash
# Full curriculum: easy → medium → hard
python train.py

# Single task
python train.py --task hard --episodes 2000
```

### 5. Run inference (trained DQN)

```bash
python inference.py --task hard --mode dqn --checkpoint checkpoints/best_model.pt
```

### 6. Start the REST server

```bash
uvicorn server.app:app --reload --port 8000
# API docs: http://localhost:8000/docs
curl http://127.0.0.1:8000/
```

### 7. Docker

```bash
docker build -t ai-ops-env -f server/Dockerfile .
docker run -p 8000:8000 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx" \
  ai-ops-env
```

---

## Mandatory Environment Variables

Per hackathon specification, the following variables **must** be defined before running inference:

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | OpenAI-compatible API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier for LLM inference | `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face / API bearer key | `hf_xxxx...` |

The `inference.py` script reads these at startup and uses the **OpenAI client** to call the model. If the LLM call fails, it falls back gracefully to the built-in heuristic agent.

---

## Project Structure

```
ai_ops_openenv/
├── agent/
│   └── dqn_agent.py        # DQN, Dueling network, Replay buffer
├── server/
│   ├── environment.py      # AIOps simulation (OpenEnv-compatible)
│   ├── app.py              # FastAPI REST server
│   └── Dockerfile
├── train.py                # Training loop + curriculum learning
├── inference.py            # Evaluation + heuristic baseline
├── client.py               # Python HTTP client
├── models.py               # Pydantic schemas
├── openenv.yaml            # OpenEnv spec v2
└── requirements.txt
```

---

## Results

| Task | Heuristic | DQN (trained) |
|------|-----------|---------------|
| easy | ~95% | ~99% |
| medium | ~70% | ~88% |
| hard | ~30% | ~72% |

*Trained for 1500 episodes with curriculum learning on CPU.*

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML.
- van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
