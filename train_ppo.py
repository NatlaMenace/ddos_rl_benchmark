# train_ppo.py

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.callbacks import RewardLoggingCallback
from src.data.export_processed import load_processed_dataset
from src.envs.ddos_envs import DDoSEnv



def make_env(df):
    def _init():
        return DDoSEnv(df, random_start=True)
    return _init


def train_ppo(total_timesteps: int = 300_000):
    # 1) Charger le TRAIN
    df_train = load_processed_dataset("processed_train_dataset.pkl")
    
    # 🔍 Vérifier le déséquilibre
    label_counts = df_train['Label'].value_counts()
    print(f"Distribution des labels:\n{label_counts}")
    print(f"Ratio Attack/Total: {label_counts.get('Attack', 0) / len(df_train):.2%}")

    # 2) VecEnv
    vec_env = DummyVecEnv([make_env(df_train)])

    # 3) Modèle PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,  # ✅ CHANGÉ : 3e-4 → 1e-4 (apprentissage plus lent)
        n_steps=512,  # ✅ CHANGÉ : 1024 → 512 (updates plus fréquentes)
        batch_size=32,  # ✅ CHANGÉ : 64 → 32 (batchs plus petits)
        n_epochs=20,  # ✅ CHANGÉ : 10 → 20 (plus d'epochs)
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,  # ✅ CHANGÉ : 0.5 → 0.01 (entropie modérée)
        verbose=1,
        tensorboard_log="runs/ppo_ddos",
    )

    # 4) Entraînement
    reward_callback = RewardLoggingCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=reward_callback,
    )

    # 5) Sauvegarde
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "ppo_ddos"
    model.save(model_path)
    print(f"[SAVE] Modèle PPO sauvegardé dans : {model_path}")


if __name__ == "__main__":
    train_ppo()