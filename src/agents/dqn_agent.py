from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Deque
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer_size: int = 1_000
    target_update_freq: int = 1000  # en steps
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.q_network = QNetwork(config.obs_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_network = QNetwork(config.obs_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.train_steps = 0

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """
        Politique ε-greedy : explore avec probabilité epsilon, sinon choisit argmax Q(s, a).
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.cfg.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def maybe_update(self) -> float | None:
        """
        Effectue une mise à jour DQN si suffisamment d'échantillons sont disponibles.
        Retourne la loss, ou None si pas d'update.
        """
        if len(self.replay_buffer) < self.cfg.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.cfg.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) courant
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Q cible avec réseau cible
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.cfg.gamma * max_next_q

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())