# eval_ppo.py

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from stable_baselines3 import PPO
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.data.export_processed import load_processed_dataset
from src.envs.ddos_envs import DDoSEnv


def evaluate_ppo(
    model_path: str = "models/ppo_ddos.zip",
    dataset_filename: str = "processed_test_dataset.pkl",  # <-- TEST
):
    # 1) Charger TEST
    df = load_processed_dataset(dataset_filename)

    # 2) Env séquentiel
    env = DDoSEnv(df, random_start=False)

    # 3) Charger PPO
    model = PPO.load(Path(model_path))

    all_preds = []
    all_labels = []

    obs, info = env.reset()
    done = False

    while not done:
        true_label = int(env.y[env.t])

        # ✅ CHANGÉ : deterministic=True → deterministic=False (ajoute stochasticité)
        action, _states = model.predict(obs, deterministic=False)
        action = int(action)

        all_labels.append(true_label)
        all_preds.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print("Labels vrais :", np.bincount(all_labels))
    print("Prédictions  :", np.bincount(all_preds))

    # 🔍 Diagnostic
    if (all_preds == 0).sum() == 0:
        print("\n⚠️  ALERTE : Le modèle ne prédit JAMAIS 0 (BENIGN) !")
        print("   → Vérifier la reward function")
        print("   → Augmenter l'entropie (ent_coef)")
    if (all_preds == 1).sum() == 0:
        print("\n⚠️  ALERTE : Le modèle ne prédit JAMAIS 1 (ATTACK) !")

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    print("=== Évaluation PPO (mode exploitation) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (Attack=1) : {precision:.4f}")
    print(f"Recall    (Attack=1) : {recall:.4f}")
    print(f"F1-score  (Attack=1) : {f1:.4f}")
    print("\nMatrice de confusion (ligne = vrai, colonne = prédit) :")
    print(cm)
    print("\nRapport détaillé :")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )

    out_path = Path("models") / "ppo_ddos_eval_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"y_true": all_labels.tolist(), "y_pred": all_preds.tolist()}, f, indent=2)
    print(f"\n[SAVE] Prédictions sauvegardées dans : {out_path}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "support": {
            "benign": int(np.sum(all_labels == 0)),
            "attack": int(np.sum(all_labels == 1)),
        }
    }

    out_metrics = Path("models") / "ppo_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SAVE] Metrics sauvegardées dans : {out_metrics}")


if __name__ == "__main__":
    evaluate_ppo()