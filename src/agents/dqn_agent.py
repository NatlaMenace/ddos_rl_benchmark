# src/agents/dqn_agent.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.q_network import QNetwork, build_q_network_from_env
from src.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Agent DQN avec :
      - epsilon-greedy
      - replay buffer
      - target network mis à jour périodiquement
    """

    def __init__(
        self,
        env,
        hidden_sizes: Sequence[int] = (128, 128),
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        target_update_freq: int = 1_000,
        device: Optional[str] = None,
    ) -> None:

        self.env = env

        # Dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Réseaux Q
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)

        self.target_q_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)

        # Initialisation : target = online
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Hyperparamètres RL
        self.gamma = gamma
        self.batch_size = batch_size

        # Epsilon-greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # en nombre de steps
        self.total_steps = 0

        # Target network
        self.target_update_freq = target_update_freq

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=self.state_dim,
            batch_size=batch_size,
        )

        # Pour le suivi
        self.loss_fn = nn.MSELoss()
        self.last_loss = None  # pour logger la dernière loss observée

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------

    def _get_epsilon(self) -> float:
        """
        Planning linéaire de epsilon :
        - commence à epsilon_start
        - décroît jusqu'à epsilon_end
        - sur epsilon_decay steps
        """
        frac = min(1.0, self.total_steps / self.epsilon_decay)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Retourne une action epsilon-greedy à partir de l'état courant.
        """
        eps = self._get_epsilon() if explore else 0.0

        if explore and np.random.rand() < eps:
            # Action aléatoire
            return self.env.action_space.sample()

        # Action greedy : argmax_a Q(s,a)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)  # (1, action_dim)
        action = int(q_values.argmax(dim=1).item())
        return action

    # ------------------------------------------------------------------
    # Stockage d'une transition
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.store(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Une étape d'apprentissage (update du réseau Q)
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Effectue une update de DQN si suffisamment de données sont disponibles.
        Retourne la loss (float) ou None si pas d'update.
        """

        if not self.buffer.can_sample():
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.buffer.sample()

        states_t = torch.from_numpy(states).float().to(self.device)           # (B, state_dim)
        actions_t = torch.from_numpy(actions).long().to(self.device)          # (B,)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)         # (B,)
        next_states_t = torch.from_numpy(next_states).float().to(self.device) # (B, state_dim)
        dones_t = torch.from_numpy(dones).float().to(self.device)             # (B,)

        # Q(s,a) courant
        q_values = self.q_net(states_t)                      # (B, action_dim)
        q_values_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Q-target(s',a') via target network
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states_t)   # (B, action_dim)
            max_next_q_values, _ = next_q_values.max(dim=1)    # (B,)
            targets = rewards_t + self.gamma * (1.0 - dones_t) * max_next_q_values

        loss = self.loss_fn(q_values_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.last_loss = loss.item()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network sync
    # ------------------------------------------------------------------

    def maybe_update_target(self) -> None:
        """
        Met à jour le target network tous les target_update_freq steps.
        """
        if self.total_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Boucle d'interaction sur un step (convenience pour la suite)
    # ------------------------------------------------------------------

    def interact_and_learn(self, state: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """
        Effectue :
         - sélection d'action
         - step dans l'env
         - stockage de la transition
         - étape d'apprentissage
         - update éventuelle du target network

        Retourne (next_state, reward, done)
        """

        action = self.select_action(state, explore=True)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.store_transition(state, action, reward, next_state, done)

        self.total_steps += 1

        loss = self.train_step()
        self.maybe_update_target()

        return next_state, reward, done