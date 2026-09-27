from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .simulation import simulate_euler_maruyama
from .weak_fpe import weak_fpe_discrepancy


@dataclass
class TrainerConfig:
    dt: float = 0.01
    sigma: float = 1.0
    learning_rate_drift: float = 1e-4
    learning_rate_critic: float = 1e-4
    critic_steps: int = 5


class AggregateDynamicsTrainer:
    """Alternating W1 / weak-FPE trainer for aggregate snapshots.

    observation_steps maps each observed aggregate snapshot to the corresponding
    Euler-Maruyama time index measured from the initial snapshot.
    """

    def __init__(
        self,
        drift,
        critics: Sequence[torch.nn.Module],
        observation_steps: Sequence[int],
        config: TrainerConfig,
    ):
        if len(critics) != len(observation_steps):
            raise ValueError("one critic is required for each observed target snapshot")

        self.drift = drift
        self.critics = list(critics)
        self.observation_steps = list(observation_steps)
        self.config = config

        self.drift_optimizer = torch.optim.Adam(
            drift.parameters(),
            lr=config.learning_rate_drift,
        )
        self.critic_optimizers = [
            torch.optim.Adam(c.parameters(), lr=config.learning_rate_critic)
            for c in self.critics
        ]

    @staticmethod
    def _requires_grad(module, enabled: bool):
        for p in module.parameters():
            p.requires_grad_(enabled)

    def _generate_path(self, initial_samples):
        max_step = max(self.observation_steps)
        return simulate_euler_maruyama(
            initial_samples,
            self.drift,
            max_step,
            self.config.dt,
            self.config.sigma,
        )

    def step(
        self,
        initial_samples: torch.Tensor,
        observed_snapshots: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        if len(observed_snapshots) != len(self.critics):
            raise ValueError("observed_snapshots must match critics")

        cfg = self.config

        # ---------------- Critic maximization ----------------
        self._requires_grad(self.drift, False)
        for critic in self.critics:
            self._requires_grad(critic, True)

        critic_values = []

        for _ in range(cfg.critic_steps):
            path = self._generate_path(initial_samples)

            for critic, optimizer, observed, end_step in zip(
                self.critics,
                self.critic_optimizers,
                observed_snapshots,
                self.observation_steps,
            ):
                optimizer.zero_grad(set_to_none=True)

                discrepancy = weak_fpe_discrepancy(
                    critic,
                    self.drift,
                    initial_samples,
                    observed,
                    path,
                    cfg.dt,
                    cfg.sigma,
                    end_step,
                )

                # maximize discrepancy
                (-discrepancy).backward()
                optimizer.step()
                critic_values.append(float(discrepancy.detach()))

        # ---------------- Drift minimization ----------------
        self._requires_grad(self.drift, True)
        for critic in self.critics:
            self._requires_grad(critic, False)

        self.drift_optimizer.zero_grad(set_to_none=True)
        path = self._generate_path(initial_samples)

        discrepancies = []
        for critic, observed, end_step in zip(
            self.critics,
            observed_snapshots,
            self.observation_steps,
        ):
            discrepancies.append(
                weak_fpe_discrepancy(
                    critic,
                    self.drift,
                    initial_samples,
                    observed,
                    path,
                    cfg.dt,
                    cfg.sigma,
                    end_step,
                )
            )

        drift_loss = torch.stack(discrepancies).sum()
        drift_loss.backward()
        self.drift_optimizer.step()

        return {
            "drift_loss": float(drift_loss.detach()),
            "mean_critic_discrepancy": (
                float(sum(critic_values) / len(critic_values))
                if critic_values else float("nan")
            ),
        }
