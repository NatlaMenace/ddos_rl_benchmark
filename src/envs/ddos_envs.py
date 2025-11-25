# src/envs/ddos_env.py

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# ---------------------------------------------------------
# 🔧 Utilitaires (chemins)
# ---------------------------------------------------------

def get_processed_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def load_selected_features() -> List[str]:
    """Charge la liste des features sélectionnées (top-k)."""
    path = get_processed_dir() / "selected_features.json"
    with open(path, "r") as f:
        return json.load(f)


def load_scaler():
    """Charge le scaler sauvegardé."""
    path = get_processed_dir() / "scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------
# 🚀 DDoS Environment (fenêtre séquentielle)
# ---------------------------------------------------------

class DDoSEnv(gym.Env):
    """
    Environnement RL pour détection DDoS.
    Observation = fenêtre glissante de W flux normalisés (flattened).

    Actions :
        0 = BENIGN
        1 = ATTACK
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 32,
        label_col: str = "Label",
        random_start=True,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.label_col = label_col
        self.random_start = random_start 

        # Charger infos preprocessing
        self.feature_cols = load_selected_features()
        self.scaler = load_scaler()

        # Préparer données normalisées (DataFrame → ndarray)
        X = self.scaler.transform(self.df[self.feature_cols].values)
        self.X = X.astype(np.float32)

        # Labels convertis en 0 ou 1 (attaque ou normal)
        labels = (
            self.df[self.label_col]
            .astype(str)
            .str.strip()        # enlève espaces
            .str.lower()        # tout en minuscule
        )
        self.y = (labels != "benign").astype(int).values
        # Compteur temporel
        self.t = 0

        # Observation_space = vector flattené
        obs_dim = self.window_size * len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action_space = binaire (normal / attaque)
        self.action_space = spaces.Discrete(2)

        # Buffer FIFO
        self.window_buffer = np.zeros(
            (self.window_size, len(self.feature_cols)), dtype=np.float32
        )

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            # Commencer à un index aléatoire
            self.t = np.random.randint(self.window_size, len(self.X) - 1)
        else:
            # Parcours séquentiel pour l'évaluation
            self.t = self.window_size

        start = self.t - self.window_size
        self.window_buffer[:] = self.X[start:self.t]

        return self._get_obs(), {}


    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

    def step(self, action: int):
        assert self.action_space.contains(action)

        # Vérité terrain du flux courant
        y_t = int(self.y[self.t])

        # Reward
        reward = self._compute_reward(action, y_t)

        # Passage au prochain flux
        self.t += 1
        terminated = (self.t >= len(self.X) - 1)

        # Update de la fenêtre glissante
        self.window_buffer[:-1] = self.window_buffer[1:]
        self.window_buffer[-1] = self.X[self.t]

        return self._get_obs(), reward, terminated, False, {}

    # ---------------------------------------------------------
    # OBSERVATION BUILDER
    # ---------------------------------------------------------

    def _get_obs(self):
        return self.window_buffer.flatten()

    # ---------------------------------------------------------
    # REWARD
    # ---------------------------------------------------------

    def _compute_reward(self, action: int, y_t: int) -> float:
        """
        Reward équilibrée pour éviter le class collapse.
        y_t : 0 = normal, 1 = attack
        action : 0 = normal, 1 = attack
        """

        if action == y_t:
            # Prédiction correcte (TP ou TN)
            return 1.0
        else:
            # Erreur (même pénalité pour FP et FN)
            return -1.0  # ✅ CHANGÉ : pénalité uniforme

    # ---------------------------------------------------------
    # RENDER (optionnel)
    # ---------------------------------------------------------

    def render(self):
        print(f"t={self.t} | true={self.y[self.t]} | buf_mean={self.window_buffer.mean():.4f}")