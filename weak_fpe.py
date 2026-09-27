from __future__ import annotations

import torch


def grad_and_laplacian(critic, x: torch.Tensor):
    """Return grad f(x) and Laplacian f(x) for a scalar critic."""
    x = x.detach().clone().requires_grad_(True)
    f = critic(x)

    grad = torch.autograd.grad(
        f,
        x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]

    laplacian = torch.zeros(
        x.shape[0], 1, dtype=x.dtype, device=x.device
    )

    for d in range(x.shape[1]):
        second = torch.autograd.grad(
            grad[:, d:d+1],
            x,
            grad_outputs=torch.ones_like(grad[:, d:d+1]),
            create_graph=True,
            retain_graph=True,
        )[0][:, d:d+1]
        laplacian = laplacian + second

    return grad, laplacian


def generator_operator(
    critic,
    drift,
    x: torch.Tensor,
    sigma: float,
):
    """Weak Fokker-Planck generator L_g f.

    L_g f = g(x) dot grad f(x) + 0.5 sigma^2 Delta f(x)
    """
    x_req = x.detach().clone().requires_grad_(True)
    grad_f, lap_f = grad_and_laplacian(critic, x_req)
    g = drift(x_req)
    drift_term = (g * grad_f).sum(dim=1, keepdim=True)
    return drift_term + 0.5 * (sigma ** 2) * lap_f


def trapezoidal_operator_integral(
    critic,
    drift,
    path: torch.Tensor,
    dt: float,
    sigma: float,
    end_step: int,
):
    """Approximate integral_0^t E[L_g f(X_s)] ds by trapezoidal rule.

    path shape: [batch, steps + 1, state_dim]
    end_step is the inclusive terminal time index.
    """
    if end_step < 1 or end_step >= path.shape[1]:
        raise ValueError("end_step must be between 1 and path.shape[1]-1")

    values = []
    for step in range(end_step + 1):
        op = generator_operator(
            critic,
            drift,
            path[:, step, :],
            sigma,
        ).mean()
        values.append(op)

    if end_step == 1:
        return 0.5 * dt * (values[0] + values[1])

    middle = torch.stack(values[1:-1]).sum()
    return dt * (0.5 * values[0] + middle + 0.5 * values[-1])


def weak_fpe_discrepancy(
    critic,
    drift,
    initial_samples: torch.Tensor,
    observed_samples: torch.Tensor,
    generated_path: torch.Tensor,
    dt: float,
    sigma: float,
    end_step: int,
):
    """Paper-style weak-FPE/Wasserstein discrepancy for one observation time."""
    f_observed = critic(observed_samples).mean()
    f_initial = critic(initial_samples).mean()

    integral = trapezoidal_operator_integral(
        critic,
        drift,
        generated_path,
        dt,
        sigma,
        end_step,
    )

    return f_observed - f_initial - integral
