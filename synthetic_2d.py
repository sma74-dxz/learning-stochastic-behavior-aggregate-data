"""Compact 2-D aggregate-dynamics demonstration.

This is intentionally smaller than the full historical paper run.
"""

import json
from pathlib import Path

import numpy as np
import torch

from aggregate_dynamics.models import DriftNet, WassersteinCritic
from aggregate_dynamics.simulation import simulate_euler_maruyama
from aggregate_dynamics.trainer import AggregateDynamicsTrainer, TrainerConfig


def true_drift(x):
    # Stable linear system used only to generate demonstration snapshots.
    A = torch.tensor([[4.0, 0.0], [0.0, 1.0]], device=x.device)
    b = torch.tensor([-12.0, -12.0], device=x.device)
    return -(x @ A.T + b)


@torch.no_grad()
def generate_true_path(initial, steps, dt, sigma):
    x = initial.clone()
    path = [x.clone()]
    for _ in range(steps):
        x = x + dt * true_drift(x) + sigma * (dt ** 0.5) * torch.randn_like(x)
        path.append(x.clone())
    return torch.stack(path, dim=1)


def main():
    cfg = json.loads(Path("configs/synthetic_2d.json").read_text())

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = cfg["samples"]
    initial = torch.randn(n, 2, device=device)

    max_step = max(cfg["observation_steps"])
    true_path = generate_true_path(
        initial,
        max_step,
        cfg["dt"],
        cfg["sigma"],
    )

    snapshots = [
        true_path[:, step, :].clone()
        for step in cfg["observation_steps"]
    ]

    drift = DriftNet(
        state_dim=2,
        hidden_dim=cfg["hidden_width"],
        hidden_layers=1,
        activation="tanh",
    ).to(device)

    critics = [
        WassersteinCritic(
            state_dim=2,
            hidden_dim=cfg["hidden_width"],
            hidden_layers=3,
        ).to(device)
        for _ in cfg["observation_steps"]
    ]

    trainer = AggregateDynamicsTrainer(
        drift,
        critics,
        cfg["observation_steps"],
        TrainerConfig(
            dt=cfg["dt"],
            sigma=cfg["sigma"],
            learning_rate_drift=1e-4,
            learning_rate_critic=1e-4,
            critic_steps=cfg["critic_steps"],
        ),
    )

    for iteration in range(cfg["iterations"]):
        metrics = trainer.step(initial, snapshots)
        if iteration % 25 == 0 or iteration == cfg["iterations"] - 1:
            print(iteration, metrics)

    learned_path = simulate_euler_maruyama(
        initial,
        drift,
        max_step,
        cfg["dt"],
        cfg["sigma"],
    )

    mse = torch.mean(
        (learned_path[:, max_step, :].mean(dim=0)
         - true_path[:, max_step, :].mean(dim=0)) ** 2
    ).item()

    print("terminal mean-vector MSE:", mse)


if __name__ == "__main__":
    main()
