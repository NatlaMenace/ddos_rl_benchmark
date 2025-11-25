# train_dqn.py

from __future__ import annotations

from pathlib import Path
import json

from src.data.export_processed import load_processed_dataset
from src.envs.ddos_envs import DDoSEnv
from src.agents.dqn_agent import DQNAgent


def train_dqn(
    num_episodes: int = 300,
    max_steps_per_episode: int = 1_000,
):
    # 1) Charger le dataset prétraité TRAIN
    df_train = load_processed_dataset("processed_train_dataset.pkl")

    # 2) Créer l'environnement RL
    env = DDoSEnv(df_train, random_start=True)

    # 3) Initialiser l'agent DQN
    agent = DQNAgent(env)

    episode_rewards = []
    episode_losses = []  # <-- nouvelle liste

    # 4) Boucle d'entraînement
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # pour moyenner la loss sur l'épisode
        losses_this_episode = []

        while not done and steps < max_steps_per_episode:
            # ta méthode actuelle : ne pas changer la signature
            next_state, reward, done = agent.interact_and_learn(state)

            state = next_state
            total_reward += reward
            steps += 1

            # si l'agent a fait un update, last_loss a été mis à jour
            if agent.last_loss is not None:
                losses_this_episode.append(agent.last_loss)

        episode_rewards.append(total_reward)

        # moyenne de loss sur l'épisode (ou None si aucun update)
        if len(losses_this_episode) > 0:
            mean_loss = float(sum(losses_this_episode) / len(losses_this_episode))
        else:
            mean_loss = None
        episode_losses.append(mean_loss)

        if episode % 10 == 0 or episode == 1:
            eps = agent._get_epsilon()
            print(
                f"[Episode {episode}/{num_episodes}] "
                f"Reward = {total_reward:.2f} | "
                f"Mean loss = {mean_loss:.4f} | "
                f"Epsilon = {eps:.3f} | "
                f"Buffer size = {agent.buffer.size}"
            )

    # 5) Sauvegarde modèle + courbes
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # modèle
    model_path = models_dir / "dqn_ddos.pt"
    agent.q_net.cpu()
    from torch import save as torch_save
    torch_save(agent.q_net.state_dict(), model_path)

    # courbes reward + loss
    curves_path = models_dir / "dqn_training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(
            {
                "episode_rewards": episode_rewards,
                "episode_losses": episode_losses,
            },
            f,
            indent=2,
        )

    print(f"[SAVE] Modèle DQN : {model_path}")
    print(f"[SAVE] Courbes DQN (reward/loss) : {curves_path}")


if __name__ == "__main__":
    train_dqn()