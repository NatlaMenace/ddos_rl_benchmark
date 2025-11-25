# src/agents/replay_buffer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ReplayBuffer:
    """
    Buffer de rejouement pour DQN.

    Stocke des transitions (s, a, r, s', done) et permet d'échantillonner
    des mini-batches pour l'apprentissage.
    """

    capacity: int
    state_dim: int
    batch_size: int

    def __post_init__(self):
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0           # index d'insertion
        self.size = 0          # nombre actuel d'éléments

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr

        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self) -> bool:
        return self.size >= self.batch_size

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne un mini-batch (s, a, r, s', done) de taille batch_size.
        """
        if not self.can_sample():
            raise ValueError("Pas assez d'échantillons dans le buffer pour échantillonner.")

        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        batch_states = self.states[idxs]
        batch_actions = self.actions[idxs]
        batch_rewards = self.rewards[idxs]
        batch_next_states = self.next_states[idxs]
        batch_dones = self.dones[idxs]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        )
    
