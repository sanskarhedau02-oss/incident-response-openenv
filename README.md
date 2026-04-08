---
title: AI Ops OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🤖 AI Ops OpenEnv — PyTorch Hackathon Submission

> **Scaler School of Technology | PyTorch Hackathon**  
> Autonomous incident-response agent trained with Deep Q-Networks

---

## Overview

**AI Ops OpenEnv** is a production-grade simulation environment where a PyTorch-powered RL agent learns to autonomously remediate cloud infrastructure incidents — restarting failed services, scaling compute, rolling back bad deploys, and flushing caches — all without human intervention.

The environment follows the **OpenEnv** specification and is compatible with any agent that speaks HTTP or the Python client.

---

## API Endpoints

Once running, visit `/docs` for the full interactive Swagger UI.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Liveness check |
| GET | `/info` | Environment metadata |
| POST | `/reset?task=easy` | Start a new episode |
| POST | `/step` | Apply an action |
| GET | `/state` | Current raw state |
| GET | `/observation` | Normalised 12-float vector |

---

## Mandatory Environment Variables

Set these in your Space's **Settings → Variables and secrets**:

| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Hugging Face / API bearer key |

---

## Project Structure

```
├── agent/
│   ├── __init__.py
│   └── dqn_agent.py
├── server/
│   ├── __init__.py
│   ├── environment.py
│   ├── app.py
│   └── models.py
├── train.py
├── inference.py
├── client.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML.
- van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
