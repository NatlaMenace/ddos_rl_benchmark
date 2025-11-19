import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

from stable_baselines3.common.env_util import make_vec_env

from src.envs.ddos_env import DDoSDatasetEnv
from src.agents.ppo_agent import PPOConfig, make_ppo_model


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement PPO sur CIC-DDoS2019 (DDoSDatasetEnv).")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Nombre total de pas d'entraînement")
    parser.add_argument("--n-envs", type=int, default=1, help="Nombre d'environnements parallèles")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps par épisode")
    parser.add_argument("--device", type=str, default="cpu", help="cpu ou cuda")
    parser.add_argument("--out-dir", type=str, default="models/ppo", help="Répertoire de sauvegarde du modèle")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Nombre d'épisodes pour l'évaluation")
    return parser.parse_args()


def evaluate_ppo(model, max_steps: int, n_episodes: int = 10):
    """
    Évalue le modèle PPO sur l'environnement de test (split='test').

    Retourne :
    - rewards_episodes : liste des rewards par épisode
    - y_true : labels réels (concatenés)
    - y_pred : actions prédites (concatenées)
    """
    env = DDoSDatasetEnv(split="test", max_steps=max_steps)

    rewards_episodes = []
    all_true = []
    all_pred = []

    for _ in trange(n_episodes, desc="Évaluation PPO"):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            ep_reward += reward
            all_true.append(info["true_label"])
            all_pred.append(int(action))

        rewards_episodes.append(ep_reward)

    env.close()
    return np.array(rewards_episodes), np.array(all_true), np.array(all_pred)


def save_confusion_matrix_figure(cm: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion - PPO")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_markdown_report(report_str: str, cm: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = "# Rapport PPO\n\n"
    md += "## Rapport de classification\n\n"
    md += "```\n" + report_str + "\n```\n\n"
    md += "## Matrice de confusion\n\n"
    md += "|       | Pred 0 | Pred 1 |\n"
    md += "|-------|--------|--------|\n"
    md += f"| Label 0 | {cm[0][0]} | {cm[0][1]} |\n"
    md += f"| Label 1 | {cm[1][0]} | {cm[1][1]} |\n"

    out_path.write_text(md, encoding="utf-8")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Création d'un env vectorisé pour PPO (split train)
    train_env = make_vec_env(
        lambda: DDoSDatasetEnv(split="train", max_steps=args.max_steps),
        n_envs=args.n_envs,
    )

    cfg = PPOConfig(device=args.device)
    model = make_ppo_model(train_env, cfg)

    print(f"[PPO] Entraînement sur {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)
    train_env.close()

    # Sauvegarde du modèle PPO
    model_path = out_dir / "ppo_cicddos"
    model.save(str(model_path))
    print(f"[SAVE] Modèle PPO sauvegardé dans {model_path.with_suffix('.zip').resolve()}")

    # Évaluation sur le split test
    rewards_episodes, y_true, y_pred = evaluate_ppo(
        model,
        max_steps=args.max_steps,
        n_episodes=args.eval_episodes,
    )

    # Sauvegarde des rewards par épisode
    np.save(out_dir / "episode_rewards.npy", rewards_episodes)

    # Rapport de classification
    report_str = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[PPO] Rapport de classification (test) :")
    print(report_str)
    print("[PPO] Matrice de confusion :")
    print(cm)

    # Sauvegarde rapport + matrice de confusion
    reports_dir = Path("reports")
    save_markdown_report(report_str, cm, reports_dir / "ppo_report.md")
    save_confusion_matrix_figure(cm, reports_dir / "ppo_confusion_matrix.png")

    print(f"[SAVE] Rapport PPO sauvegardé dans {reports_dir / 'ppo_report.md'}")
    print(f"[SAVE] Matrice de confusion PPO sauvegardée dans {reports_dir / 'ppo_confusion_matrix.png'}")


if __name__ == "__main__":
    main()