"""
DQN Agent for AI Ops Incident Response
========================================
Architecture:
  - Deep Q-Network (DQN) with experience replay and target network
  - Dueling DQN head: separates Value V(s) and Advantage A(s,a)
  - Double DQN update: decorrelates action selection from evaluation
  - ε-greedy exploration with linear annealing

Reference: Mnih et al. (2015), Wang et al. (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    obs_dim: int = 12
    n_actions: int = 6
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99          # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    replay_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 200  # steps between target-net hard updates
    min_replay_size: int = 256     # warm-up before learning starts
    grad_clip: float = 10.0


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class DuelingDQN(nn.Module):
    """
    Dueling DQN with shared encoder + separate Value and Advantage streams.
    Q(s,a) = V(s) + A(s,a) − mean_a'[A(s,a')]
    """

    def __init__(self, cfg: DQNConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        V = self.value_stream(feat)
        A = self.advantage_stream(feat)
        Q = V + A - A.mean(dim=-1, keepdim=True)
        return Q


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self._buf.append((
            np.array(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.stack(obs)),
            torch.tensor(acts),
            torch.tensor(rews),
            torch.tensor(np.stack(next_obs)),
            torch.tensor(dones),
        )

    def __len__(self):
        return len(self._buf)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    Double-DQN + Dueling architecture agent.

    Usage:
        agent = DQNAgent(cfg)
        action_idx = agent.select_action(obs_vector)
        agent.store_transition(obs, action_idx, reward, next_obs, done)
        loss = agent.learn()           # returns None until replay is warm
    """

    # Maps discrete action index → (action_type, target)
    ACTION_MAP = [
        {"action_type": "noop",            "target": None},
        {"action_type": "restart_service", "target": "auth-service"},
        {"action_type": "restart_service", "target": "api"},
        {"action_type": "scale_up",        "target": None},
        {"action_type": "rollback",        "target": None},
        {"action_type": "flush_cache",     "target": None},
    ]

    def __init__(self, cfg: DQNConfig = None, device: str = "cpu"):
        self.cfg = cfg or DQNConfig()
        self.device = torch.device(device)

        self.online_net = DuelingDQN(self.cfg).to(self.device)
        self.target_net = DuelingDQN(self.cfg).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(self.cfg.replay_capacity)

        self._total_steps: int = 0
        self._epsilon: float = self.cfg.epsilon_start

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------
    def select_action(self, obs: list[float]) -> int:
        """ε-greedy action selection. Returns action index."""
        self._update_epsilon()
        if random.random() < self._epsilon:
            return random.randrange(self.cfg.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(obs_t)
        return int(q_vals.argmax(dim=-1).item())

    def action_to_dict(self, action_idx: int) -> dict:
        """Convert discrete action index to environment-compatible dict."""
        return dict(self.ACTION_MAP[action_idx])

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay.push(obs, action, reward, next_obs, done)
        self._total_steps += 1
        if self._total_steps % self.cfg.target_update_freq == 0:
            self._hard_update_target()

    def learn(self) -> float | None:
        """Sample a minibatch and perform one gradient update. Returns loss."""
        if len(self.replay) < self.cfg.min_replay_size:
            return None

        obs, acts, rews, next_obs, dones = self.replay.sample(self.cfg.batch_size)
        obs      = obs.to(self.device)
        acts     = acts.to(self.device)
        rews     = rews.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        # Double DQN: online net selects action, target net evaluates
        with torch.no_grad():
            next_actions = self.online_net(next_obs).argmax(dim=-1)
            next_q = self.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rews + self.cfg.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
            "epsilon": self._epsilon,
            "config": self.cfg,
        }, path)
        print(f"[Agent] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_steps = ckpt["total_steps"]
        self._epsilon = ckpt["epsilon"]
        print(f"[Agent] Loaded checkpoint from {path}")

    @property
    def epsilon(self):
        return self._epsilon

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------
    def _update_epsilon(self):
        frac = min(self._total_steps / self.cfg.epsilon_decay_steps, 1.0)
        self._epsilon = self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
