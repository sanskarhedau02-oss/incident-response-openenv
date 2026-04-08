"""
train.py — DQN training loop for AIOps incident-response agent
================================================================

Features
--------
* Curriculum learning: starts on 'easy', promotes to 'medium'/'hard'
  when the agent achieves a sustained success rate.
* Metrics tracked per episode: total_reward, steps, success, task level.
* Periodic evaluation runs (deterministic, ε=0) to measure true policy quality.
* Checkpoint saved whenever eval score improves (best-model tracking).
* Console progress with clean tabular output — no external deps needed.

Usage
-----
    python train.py                        # train all tasks, default config
    python train.py --task hard            # single-task training
    python train.py --episodes 2000 --eval-every 100

Requires: torch, numpy  (see requirements.txt)
"""

import argparse
import json
import os
import sys
import time
from statistics import mean

# Make sure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import AIOpsEnvironment, MAX_STEPS
from agent.dqn_agent import DQNAgent, DQNConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(agent: DQNAgent, task: str, n_episodes: int = 20) -> dict:
    """Run n_episodes with ε=0 (greedy) and return aggregate stats."""
    saved_eps = agent.epsilon
    agent._epsilon = 0.0  # force greedy

    env = AIOpsEnvironment(task=task, stochastic=False)
    successes, rewards, step_counts = [], [], []

    for _ in range(n_episodes):
        obs_dict = env.reset()
        obs = env.get_observation_vector()
        ep_reward = 0.0

        for _ in range(MAX_STEPS):
            action_idx = agent.select_action(obs)
            action = agent.action_to_dict(action_idx)
            _, reward, done, _ = env.step(action)
            obs = env.get_observation_vector()
            ep_reward += reward
            if done:
                break

        successes.append(env._is_done())
        rewards.append(ep_reward)
        step_counts.append(env._step_count)

    agent._epsilon = saved_eps  # restore
    return {
        "task": task,
        "success_rate": mean(successes),
        "mean_reward": round(mean(rewards), 4),
        "mean_steps": round(mean(step_counts), 1),
    }


def _fmt_row(ep, eps, task, ep_rew, loss, eval_sr=None):
    eval_str = f"  ✓ eval={eval_sr:.0%}" if eval_sr is not None else ""
    return (
        f"  Ep {ep:>5d} | ε={eps:.3f} | task={task:<6s} | "
        f"rew={ep_rew:>+7.3f} | loss={loss if loss else '    ---':>8}{eval_str}"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    task: str = "curriculum",
    total_episodes: int = 1500,
    eval_every: int = 100,
    save_dir: str = "checkpoints",
    device: str = "cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    metrics_log = []

    cfg = DQNConfig(
        obs_dim=12,
        n_actions=6,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=int(total_episodes * 8),
        replay_capacity=15_000,
        batch_size=64,
        target_update_freq=300,
        min_replay_size=512,
    )
    agent = DQNAgent(cfg=cfg, device=device)

    # Curriculum: task sequence and promotion thresholds
    task_seq = ["easy", "medium", "hard"] if task == "curriculum" else [task]
    task_idx = 0
    current_task = task_seq[task_idx]
    recent_successes = []   # rolling window for promotion

    best_eval_score = -float("inf")
    best_ckpt = os.path.join(save_dir, "best_model.pt")
    last_loss = None

    print("\n" + "=" * 72)
    print("  AIOps DQN Training — Double Dueling DQN + Curriculum Learning")
    print("=" * 72)
    print(f"  Device:   {device}")
    print(f"  Task:     {task}  →  sequence: {task_seq}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Config:   lr={cfg.lr}, γ={cfg.gamma}, batch={cfg.batch_size}")
    print("=" * 72 + "\n")

    t0 = time.time()

    for ep in range(1, total_episodes + 1):
        env = AIOpsEnvironment(task=current_task, stochastic=True)
        obs_dict = env.reset()
        obs = env.get_observation_vector()
        ep_reward = 0.0
        ep_loss_vals = []

        for _ in range(MAX_STEPS):
            action_idx = agent.select_action(obs)
            action = agent.action_to_dict(action_idx)
            _, reward, done, _ = env.step(action)
            next_obs = env.get_observation_vector()

            agent.store_transition(obs, action_idx, reward, next_obs, float(done))
            loss = agent.learn()
            if loss is not None:
                ep_loss_vals.append(loss)
                last_loss = loss

            obs = next_obs
            ep_reward += reward
            if done:
                break

        success = env._is_done()
        recent_successes.append(float(success))
        if len(recent_successes) > 50:
            recent_successes.pop(0)

        avg_loss = round(mean(ep_loss_vals), 5) if ep_loss_vals else None

        # Curriculum promotion
        if task == "curriculum" and task_idx < len(task_seq) - 1:
            if len(recent_successes) >= 30 and mean(recent_successes) >= 0.75:
                task_idx += 1
                current_task = task_seq[task_idx]
                recent_successes.clear()
                print(f"\n  🎓 Promoted to task: [{current_task.upper()}]  (ep={ep})\n")

        # Periodic evaluation
        eval_result = None
        if ep % eval_every == 0:
            eval_result = evaluate(agent, current_task)
            score = eval_result["success_rate"]
            metrics_log.append({"ep": ep, "task": current_task, **eval_result})
            if score > best_eval_score:
                best_eval_score = score
                agent.save(best_ckpt)

        # Logging
        if ep % 10 == 0 or eval_result:
            sr = eval_result["success_rate"] if eval_result else None
            print(_fmt_row(ep, agent.epsilon, current_task, ep_reward, avg_loss, sr))

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best eval success rate: {best_eval_score:.0%}  → {best_ckpt}")
    print(f"{'='*72}\n")

    # Save metrics
    metrics_path = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    return agent, metrics_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on AIOps environment")
    parser.add_argument("--task", default="curriculum",
                        choices=["easy", "medium", "hard", "curriculum"])
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(
        task=args.task,
        total_episodes=args.episodes,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        device=args.device,
    )
