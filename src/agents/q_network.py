# src/agents/q_network.py

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Réseau Q approché par un MLP.
    
    - Entrée  : vecteur d'état (dimension = state_dim)
    - Sortie  : Q(s, a) pour chaque action (dimension = action_dim)
    - Architecture :
        input -> Linear -> ReLU -> Linear -> ReLU -> Linear(output)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()

        layers = []
        input_dim = state_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        # Couche de sortie : un Q-value par action
        layers.append(nn.Linear(input_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : tensor de forme (batch_size, state_dim)
        retourne : tensor (batch_size, action_dim) avec Q(s, a)
        """
        return self.net(x)


def build_q_network_from_env(
    env,
    hidden_sizes: Sequence[int] = (128, 128),
) -> QNetwork:
    """
    Helper pratique pour créer un QNetwork à partir d'un environnement Gym.
    
    - state_dim  = env.observation_space.shape[0]
    - action_dim = env.action_space.n
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    return QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

