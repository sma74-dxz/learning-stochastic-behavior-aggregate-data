from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _xavier_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DriftNet(nn.Module):
    """Time-independent drift field g_theta(x)."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        hidden_layers: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()

        activation_layer = nn.Tanh if activation == "tanh" else nn.ReLU

        layers = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), activation_layer()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, state_dim))

        self.net = nn.Sequential(*layers)
        self.apply(_xavier_init)

    def forward(self, x):
        return self.net(x)


class WassersteinCritic(nn.Module):
    """Smooth spectrally-normalized W1 test function f_theta(x).

    Tanh is used because the weak FPE objective requires first and second
    derivatives with respect to the state.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        hidden_layers: int = 3,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim

        for _ in range(hidden_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            layers += [spectral_norm(linear), nn.Tanh()]
            in_dim = hidden_dim

        layers.append(spectral_norm(nn.Linear(in_dim, 1)))
        self.net = nn.Sequential(*layers)

        # Spectral-normalized Linear modules keep the trainable original weight
        # inside the parametrization, so zeroing biases is the only explicit
        # initialization needed here.
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)
