from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"
    verbose: int = 1


def make_ppo_model(env: VecEnv, config: PPOConfig) -> PPO:
    """
    Crée un modèle PPO (Stable-Baselines3) avec une politique MLP.
    """
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
        verbose=config.verbose,
    )
    return model