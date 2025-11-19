from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PROCESSED_DIR = Path("data/processed")


def _load_split(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les données prétraitées (X_split, y_split) depuis data/processed.
    split ∈ {'train', 'test'}.
    """
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError(f"split doit être 'train' ou 'test', reçu: {split}")

    x_path = PROCESSED_DIR / f"X_{split}.npy"
    y_path = PROCESSED_DIR / f"y_{split}.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Fichiers prétraités introuvables pour le split '{split}'. "
            f"Attendus: {x_path} et {y_path}. Lancer d'abord: python -m src.data.preprocessing"
        )

    X = np.load(x_path)
    y = np.load(y_path)

    return X, y


class DDoSDatasetEnv(gym.Env):
    """
    Environnement RL simple pour la détection DDoS à partir de CIC-DDoS2019.

    - Observation : vecteur de features normalisées (float32).
    - Action : 0 = trafic normal, 1 = attaque (binaire).
    - Récompense : +1 si action == label, sinon -1.
    - Épisode : parcours d'un échantillon de la base (ordre aléatoire).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        split: str = "train",
        false_negative_penalty: float = -2.0,
        false_positive_penalty: float = -1.0,
        correct_reward: float = 1.0,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.split = split
        self.false_negative_penalty = false_negative_penalty
        self.false_positive_penalty = false_positive_penalty
        self.correct_reward = correct_reward
        self.max_steps = max_steps

        # RNG Gymnasium
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Chargement des données
        self.X, self.y = _load_split(split)
        if self.X.ndim != 2:
            raise ValueError(f"X doit être 2D (n_samples, n_features), reçu {self.X.shape}")

        self.n_samples, self.n_features = self.X.shape

        # Espace d'observation : vecteur de taille n_features, valeurs normalisées ~[-3, 3] (StandardScaler)
        obs_low = np.full(self.n_features, -10.0, dtype=np.float32)
        obs_high = np.full(self.n_features, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Actions : 0 = normal, 1 = attaque
        self.action_space = spaces.Discrete(2)

        # État interne
        self._indices: np.ndarray = np.arange(self.n_samples)
        self._current_step: int = 0
        self._max_steps_episode: int = self.n_samples if max_steps is None else min(max_steps, self.n_samples)

    def _get_observation(self) -> np.ndarray:
        idx = self._indices[self._current_step]
        obs = self.X[idx].astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Réinitialise l'épisode :
        - tire un nouvel ordre aléatoire des indices ;
        - remet le compteur de steps à 0 ;
        - renvoie la première observation.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Permutation aléatoire des indices
        self._indices = self.np_random.permutation(self.n_samples)
        self._current_step = 0

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        """
        Applique l'action et renvoie (obs_next, reward, terminated, truncated, info).
        """
        # Vérification de l'action
        assert self.action_space.contains(action), f"Action invalide : {action}"

        idx = self._indices[self._current_step]
        true_label = int(self.y[idx])

        # Calcul de la récompense
        if action == true_label:
            reward = self.correct_reward
        else:
            if true_label == 1:
                # Attaque non détectée : faux négatif (plus grave)
                reward = self.false_negative_penalty
            else:
                # Trafic normal mal classé : faux positif
                reward = self.false_positive_penalty

        # Avancer le step
        self._current_step += 1

        terminated = self._current_step >= self._max_steps_episode
        truncated = False  # pas de troncature spécifique ici

        if not terminated:
            obs = self._get_observation()
        else:
            # obs quelconque à la fin, Gym ignore normalement si terminated=True
            obs = np.zeros(self.n_features, dtype=np.float32)

        info = {"true_label": true_label, "index": int(idx)}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Pour un environnement purement tabulaire/numérique, pas de rendu particulier.
        pass

    def close(self):
        pass


def make_env(split: str = "train", seed: Optional[int] = None) -> DDoSDatasetEnv:
    """
    Helper pour créer un environnement d'entraînement ou de test.
    """
    return DDoSDatasetEnv(split=split, seed=seed)