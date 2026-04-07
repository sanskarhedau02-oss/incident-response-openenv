"""
AIOpsEnvironment — a richly simulated incident-response environment.

Observation vector (12 floats, all normalised to [0, 1]):
  [0]  cpu_usage / 100
  [1]  memory_usage / 100
  [2]  error_rate          (already in [0,1])
  [3]  latency_ms / 2000
  [4]  auth_service_status   (1=healthy, 0=down)
  [5]  api_service_status
  [6]  db_service_status
  [7]  cache_service_status
  [8]  step_count / MAX_STEPS
  [9]  incident_severity / 3  (1=low, 2=medium, 3=high)
  [10] alert_count / 10
  [11] restart_budget / MAX_RESTARTS

Action space (6 discrete):
  0  noop
  1  restart_service       (uses one restart token; needs `target`)
  2  scale_up              (reduces CPU / latency)
  3  rollback              (reduces error_rate)
  4  flush_cache           (reduces latency, may temporarily raise error_rate)
  5  escalate              (calls human — big negative reward, ends episode)
"""

import random

MAX_STEPS = 20
MAX_RESTARTS = 5
OBS_DIM = 12
N_ACTIONS = 6

SERVICE_NAMES = ["auth-service", "api", "db", "cache"]

TASK_CONFIGS = {
    "easy": {
        "services": {"auth-service": "down", "api": "healthy", "db": "healthy", "cache": "healthy"},
        "cpu_usage": 35.0,
        "memory_usage": 40.0,
        "error_rate": 0.25,
        "latency_ms": 300.0,
        "incident_severity": 1,
        "alert_count": 1,
        "incident_id": "INC-001",
        "logs": ["auth-service crashed at 14:32 UTC", "health check failed 3x"],
    },
    "medium": {
        "services": {"auth-service": "healthy", "api": "healthy", "db": "healthy", "cache": "healthy"},
        "cpu_usage": 92.0,
        "memory_usage": 78.0,
        "error_rate": 0.35,
        "latency_ms": 950.0,
        "incident_severity": 2,
        "alert_count": 4,
        "incident_id": "INC-042",
        "logs": ["CPU spike detected", "p99 latency > 900ms", "OOMKilled: 2 pods"],
    },
    "hard": {
        "services": {"auth-service": "down", "api": "degraded", "db": "healthy", "cache": "down"},
        "cpu_usage": 88.0,
        "memory_usage": 91.0,
        "error_rate": 0.72,
        "latency_ms": 1800.0,
        "incident_severity": 3,
        "alert_count": 9,
        "incident_id": "INC-999",
        "logs": [
            "cascading failure: auth → api dependency broken",
            "cache eviction storm",
            "DB connection pool exhausted",
            "PagerDuty: SEV-1 declared",
        ],
    },
}


class AIOpsEnvironment:
    """
    OpenAI-Gym-style environment for AI-driven incident response.

    Compatible with the OpenEnv spec (openenv.yaml).
    """

    def __init__(self, task: str = "easy", stochastic: bool = True, seed: int = 42):
        assert task in TASK_CONFIGS, f"Unknown task '{task}'. Choose from {list(TASK_CONFIGS)}"
        self.task = task
        self.stochastic = stochastic
        self._rng = random.Random(seed)
        self.state: dict = {}
        self.restart_budget: int = MAX_RESTARTS
        self._step_count: int = 0
        self._episode_reward: float = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        cfg = TASK_CONFIGS[self.task]
        self.state = {
            "incident_id": cfg["incident_id"],
            "services": dict(cfg["services"]),
            "cpu_usage": cfg["cpu_usage"],
            "memory_usage": cfg["memory_usage"],
            "error_rate": cfg["error_rate"],
            "latency_ms": cfg["latency_ms"],
            "incident_severity": cfg["incident_severity"],
            "alert_count": cfg["alert_count"],
            "logs": list(cfg["logs"]),
            "step_count": 0,
        }
        self.restart_budget = MAX_RESTARTS
        self._step_count = 0
        self._episode_reward = 0.0
        return dict(self.state)

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Returns (observation, reward, done, info).
        `action` must have keys: action_type (str), target (str | None).
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        reward = self._apply_action(action)
        self._apply_stochastic_drift()

        self._clamp_state()
        self._step_count += 1
        self.state["step_count"] = self._step_count
        self._episode_reward += reward

        done = self._is_done()
        info = {
            "episode_reward": round(self._episode_reward, 4),
            "restart_budget": self.restart_budget,
            "action_applied": action["action_type"],
        }
        return dict(self.state), round(float(reward), 4), done, info

    def get_observation_vector(self) -> list[float]:
        """Return the normalised 12-float observation the agent actually sees."""
        s = self.state
        svc = s["services"]
        return [
            s["cpu_usage"] / 100.0,
            s["memory_usage"] / 100.0,
            s["error_rate"],
            min(s["latency_ms"] / 2000.0, 1.0),
            1.0 if svc.get("auth-service") == "healthy" else 0.0,
            1.0 if svc.get("api") == "healthy" else 0.0,
            1.0 if svc.get("db") == "healthy" else 0.0,
            1.0 if svc.get("cache") == "healthy" else 0.0,
            self._step_count / MAX_STEPS,
            s["incident_severity"] / 3.0,
            min(s["alert_count"] / 10.0, 1.0),
            self.restart_budget / MAX_RESTARTS,
        ]

    def get_state(self) -> dict:
        return dict(self.state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_action(self, action: dict) -> float:
        a_type = action.get("action_type", "noop")
        target = action.get("target")
        reward = 0.0

        if a_type == "restart_service":
            if self.restart_budget <= 0:
                reward -= 0.15  # tried to restart without budget
            elif target and target in self.state["services"]:
                prev = self.state["services"][target]
                self.state["services"][target] = "healthy"
                self.restart_budget -= 1
                self.state["logs"].append(f"Restarted {target}")
                reward += 0.35 if prev != "healthy" else -0.05  # penalise useless restart
            else:
                reward -= 0.05  # invalid target

        elif a_type == "scale_up":
            delta = self._rng.uniform(25, 40) if self.stochastic else 30
            self.state["cpu_usage"] -= delta
            self.state["latency_ms"] -= delta * 8
            self.state["memory_usage"] += 5  # new pods consume memory
            self.state["logs"].append("Scaled up compute")
            reward += 0.25

        elif a_type == "rollback":
            delta = self._rng.uniform(0.3, 0.5) if self.stochastic else 0.4
            self.state["error_rate"] -= delta
            self.state["latency_ms"] -= 200
            self.state["alert_count"] = max(0, self.state["alert_count"] - 2)
            self.state["logs"].append("Rolled back last deploy")
            reward += 0.4

        elif a_type == "flush_cache":
            self.state["latency_ms"] -= 400
            self.state["error_rate"] += 0.05  # brief spike during flush
            if self.state["services"].get("cache") == "down":
                self.state["services"]["cache"] = "healthy"
                reward += 0.2
            self.state["logs"].append("Cache flushed and rewarmed")
            reward += 0.15

        elif a_type == "escalate":
            reward -= 0.8  # humans do not like being paged for things the agent should handle
            self.state["logs"].append("⚠ Escalated to on-call engineer")

        else:  # noop or unknown
            reward -= 0.02  # tiny cost to deter padding

        # Shape reward: award healthy-state bonuses each step
        reward += self._health_bonus()
        return reward

    def _health_bonus(self) -> float:
        """Small per-step reward for each metric already in a good range."""
        bonus = 0.0
        if self.state["cpu_usage"] < 60:
            bonus += 0.03
        if self.state["error_rate"] < 0.1:
            bonus += 0.04
        if self.state["latency_ms"] < 500:
            bonus += 0.03
        if all(v == "healthy" for v in self.state["services"].values()):
            bonus += 0.05
        return bonus

    def _apply_stochastic_drift(self):
        """Simulate noisy real-world systems: metrics fluctuate slightly."""
        if not self.stochastic:
            return
        self.state["cpu_usage"] += self._rng.gauss(0, 2)
        self.state["error_rate"] += self._rng.gauss(0, 0.01)
        self.state["latency_ms"] += self._rng.gauss(0, 15)

        # Cascading failure: if api is down, error_rate tends to climb
        if self.state["services"].get("api") != "healthy":
            self.state["error_rate"] += 0.02

    def _clamp_state(self):
        self.state["cpu_usage"] = max(0.0, min(100.0, self.state["cpu_usage"]))
        self.state["memory_usage"] = max(0.0, min(100.0, self.state["memory_usage"]))
        self.state["error_rate"] = max(0.0, min(1.0, self.state["error_rate"]))
        self.state["latency_ms"] = max(50.0, self.state["latency_ms"])

    def _is_done(self) -> bool:
        if self._step_count >= MAX_STEPS:
            return True
        s = self.state
        return (
            s["cpu_usage"] < 60
            and s["error_rate"] < 0.1
            and s["latency_ms"] < 500
            and all(v == "healthy" for v in s["services"].values())
        )
