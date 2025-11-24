from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_rewards_curves(
    dqn_rewards_path: str | Path,
    ppo_rewards_path: str | Path,
    out_path: str | Path = "reports/dqn_vs_ppo_rewards.png",
):
    """
    Trace les courbes de reward cumulée par épisode pour DQN et PPO.
    """
    dqn_rewards = np.load(dqn_rewards_path)
    ppo_rewards = np.load(ppo_rewards_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(ppo_rewards, label="PPO")
    plt.xlabel("Épisode")
    plt.ylabel("Reward cumulée")
    plt.title("Courbes d'apprentissage DQN vs PPO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()