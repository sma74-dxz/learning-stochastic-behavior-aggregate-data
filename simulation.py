from __future__ import annotations

import torch


@torch.no_grad()
def simulate_euler_maruyama(
    initial_samples: torch.Tensor,
    drift,
    steps: int,
    dt: float,
    sigma: float,
):
    """Generate a sample path using Euler-Maruyama.

    The historical implementation generated sample locations outside the
    optimization graph and then evaluated the weak FPE operator on those
    sampled locations. This function preserves that research-code behavior.
    """
    x = initial_samples.detach().clone()
    path = [x.clone()]

    sqrt_dt = dt ** 0.5

    for _ in range(steps):
        g = drift(x)
        noise = sigma * sqrt_dt * torch.randn_like(x)
        x = x + dt * g + noise
        path.append(x.clone())

    return torch.stack(path, dim=1)
