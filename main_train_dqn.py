import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from src.envs.ddos_env import DDoSDatasetEnv
from src.agents.dqn_agent import DQNAgent, DQNConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement DQN sur CIC-DDoS2019 (DDoSDatasetEnv).")
    parser.add_argument("--episodes", type=int, default=200, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps par épisode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Facteur de réduction")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch pour DQN")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Taille du replay buffer")
    parser.add_argument("--min-buffer-size", type=int, default=1_000, help="Taille minimale avant update")
    parser.add_argument("--eps-start", type=float, default=1.0, help="epsilon initial (exploration)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="epsilon minimal")
    parser.add_argument("--eps-decay", type=int, default=50_000, help="nombre de steps pour décroissance epsilon")
    parser.add_argument("--device", type=str, default="cpu", help="cpu ou cuda")
    parser.add_argument("--split", type=str, default="train", help="split utilisé par l'env (train/test)")
    parser.add_argument("--out-dir", type=str, default="models/dqn", help="Répertoire de sauvegarde du modèle")
    return parser.parse_args()


def linear_epsilon(step: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    """Décroissance linéaire d'epsilon en fonction du nombre de steps."""
    frac = max(0.0, min(1.0, 1.0 - step / eps_decay))
    return eps_end + (eps_start - eps_end) * frac


def main():
    args = parse_args()

    # Création de l'env
    env = DDoSDatasetEnv(split=args.split, max_steps=args.max_steps)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    cfg = DQNConfig(
        obs_dim=obs_dim,
        n_actions=n_actions,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        device=args.device,
    )

    agent = DQNAgent(cfg)

    total_steps = 0
    episode_rewards = []
    losses = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in trange(args.episodes, desc="Entraînement DQN"):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            epsilon = linear_epsilon(total_steps, args.eps_start, args.eps_end, args.eps_decay)

            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.push_transition(state, action, reward, next_state, done)

            loss = agent.maybe_update()
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_reward += reward
            total_steps += 1

        episode_rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"\n[EPISODE {ep+1}] Reward moyen (10 derniers) : {avg_reward:.3f}")

    # Sauvegarde du modèle DQN
    model_path = out_dir / "dqn_cicddos.pt"
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"[SAVE] Modèle DQN sauvegardé dans {model_path.resolve()}")

    # Sauvegarde des stats simples
    np.save(out_dir / "episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))
    if losses:
        np.save(out_dir / "losses.npy", np.array(losses, dtype=np.float32))

    env.close()


if __name__ == "__main__":
    main()