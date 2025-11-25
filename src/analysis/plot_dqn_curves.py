# src/analysis/plot_dqn_curves.py

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dqn_curves(curves_path: str | Path = "models/dqn_training_curves.json"):
    curves_path = Path(curves_path)
    if not curves_path.exists():
        raise FileNotFoundError(f"Fichier de courbes DQN introuvable : {curves_path}")

    with open(curves_path, "r") as f:
        data = json.load(f)

    episode_rewards = np.array(data["episode_rewards"], dtype=float)
    # Certaines pertes peuvent être None au début → on les met en NaN pour les ignorer
    raw_losses = data.get("episode_losses", [])
    episode_losses = np.array(
        [np.nan if v is None else float(v) for v in raw_losses],
        dtype=float,
    )

    return episode_rewards, episode_losses


def plot_dqn_reward(
    episode_rewards: np.ndarray,
    save_dir: str | Path = "figures",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(episode_rewards) + 1)

    plt.figure()
    plt.plot(episodes, episode_rewards)
    plt.xlabel("Épisode")
    plt.ylabel("Reward total par épisode")
    plt.title("DQN – Reward par épisode")
    plt.grid(True)

    out_path = save_dir / "dqn_reward_per_episode.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVE] Courbe de reward DQN sauvegardée dans : {out_path}")


def plot_dqn_loss(
    episode_losses: np.ndarray,
    save_dir: str | Path = "figures",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(episode_losses) + 1)

    plt.figure()
    plt.plot(episodes, episode_losses)
    plt.xlabel("Épisode")
    plt.ylabel("Loss moyenne par épisode")
    plt.title("DQN – Convergence de la loss")
    plt.grid(True)

    out_path = save_dir / "dqn_loss_per_episode.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVE] Courbe de loss DQN sauvegardée dans : {out_path}")


def main():
    episode_rewards, episode_losses = load_dqn_curves()

    plot_dqn_reward(episode_rewards)
    plot_dqn_loss(episode_losses)


if __name__ == "__main__":
    main()