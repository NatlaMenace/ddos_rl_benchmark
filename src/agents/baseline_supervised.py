from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = PROCESSED_DIR / "baseline_random_forest.joblib"


def load_processed_data():
    """
    Charge les jeux de données prétraités depuis data/processed.
    """
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test


def save_confusion_matrix_figure(cm, out_path="reports/confusion_matrix.png"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion - Baseline RandomForest")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_markdown_report(classif_report: str, cm: np.ndarray, out_path="reports/baseline_report.md"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert confusion matrix to markdown table
    md = "# Rapport Baseline RandomForest\n\n"
    md += "## Rapport de classification\n\n"
    md += "```\n" + classif_report + "\n```\n\n"
    md += "## Matrice de confusion\n\n"
    md += "|       | Pred 0 | Pred 1 |\n"
    md += "|-------|--------|--------|\n"
    md += f"| Label 0 | {cm[0][0]} | {cm[0][1]} |\n"
    md += f"| Label 1 | {cm[1][0]} | {cm[1][1]} |\n"

    out_path.write_text(md, encoding="utf-8")


def train_baseline_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
):
    """
    Entraîne une baseline RandomForestClassifier sur les données prétraitées
    et évalue ses performances sur le set de test.
    """
    print("[BASELINE] Chargement des données prétraitées...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"[BASELINE] X_train : {X_train.shape}, X_test : {X_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )

    print("[BASELINE] Entraînement du RandomForest...")
    clf.fit(X_train, y_train)

    print("[BASELINE] Évaluation sur le set de test...")
    y_pred = clf.predict(X_test)

    print("\n[BASELINE] Rapport de classification :")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("[BASELINE] Matrice de confusion :")
    print(cm)

    # Rapport classification sous forme de chaîne
    classif_rep = classification_report(y_test, y_pred, digits=4)

    # Sauvegardes supplémentaires
    save_confusion_matrix_figure(cm)
    save_markdown_report(classif_rep, cm)

    # Sauvegarde du modèle
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"[BASELINE] Modèle sauvegardé sous : {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train_baseline_random_forest()