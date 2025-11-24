from pathlib import Path

import numpy as np

from src.utils.metrics import compute_classification_metrics, metrics_to_markdown_row
from src.utils.plots import plot_rewards_curves


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rewards DQN vs PPO
    dqn_rewards_path = Path("models/dqn/episode_rewards.npy")
    ppo_rewards_path = Path("models/ppo/episode_rewards.npy")

    if dqn_rewards_path.exists() and ppo_rewards_path.exists():
        plot_rewards_curves(dqn_rewards_path, ppo_rewards_path, reports_dir / "dqn_vs_ppo_rewards.png")
        print("[COMPARE] Courbe DQN vs PPO sauvegardée dans reports/dqn_vs_ppo_rewards.png")
    else:
        print("[WARN] Impossible de tracer les rewards : fichiers episode_rewards.npy manquants.")

    # 2. Chargement des prédictions déjà obtenues (à adapter selon ce que tu sauves)
    # Ici, exemple : on imagine que tu sauvegardes plus tard y_true / y_pred de chaque modèle.
    # Pour l'instant, on met des placeholders pour la structure du code.

    comparison_md = "# Comparaison DQN / PPO / Baseline\n\n"
    comparison_md += "| Modèle | Accuracy | F1-macro | F1-weighted | Precision-macro | Recall-macro |\n"
    comparison_md += "|--------|----------|---------:|------------:|----------------:|-------------:|\n"

    # TODO: quand tu auras y_true/y_pred pour chaque modèle, tu appelleras:
    # m_baseline = compute_classification_metrics(y_true_baseline, y_pred_baseline)
    # comparison_md += metrics_to_markdown_row("Baseline RF", m_baseline)
    # m_dqn = compute_classification_metrics(y_true_dqn, y_pred_dqn)
    # comparison_md += metrics_to_markdown_row("DQN", m_dqn)
    # m_ppo = compute_classification_metrics(y_true_ppo, y_pred_ppo)
    # comparison_md += metrics_to_markdown_row("PPO", m_ppo)

    # Pour l’instant on laisse le squelette :
    comparison_md += "| Baseline RF | (à remplir) | (à remplir) | (à remplir) | (à remplir) | (à remplir) |\n"
    comparison_md += "| DQN         | (à remplir) | (à remplir) | (à remplir) | (à remplir) | (à remplir) |\n"
    comparison_md += "| PPO         | 0.2240      | 0.0230      | 0.0830      | (à remplir) | (à remplir) |\n"

    out_md = reports_dir / "comparison_table.md"
    out_md.write_text(comparison_md, encoding="utf-8")
    print(f"[COMPARE] Tableau de comparaison sauvegardé dans {out_md}")


if __name__ == "__main__":
    main()