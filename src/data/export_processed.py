# src/data/export_processed.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional
import pandas as pd


# ---------------------------------------------------------
# 🔧 Chemin vers data/processed/
# ---------------------------------------------------------

def get_processed_dir() -> Path:
    """Retourne le dossier data/processed/."""
    return Path(__file__).resolve().parents[2] / "data" / "processed"


# ---------------------------------------------------------
# 💾 Sauvegarde du dataset
# ---------------------------------------------------------

def save_processed_dataset(
    df: pd.DataFrame,
    filename: str = "processed_dataset.pkl",
):
    """
    Sauvegarde du DataFrame prétraité (normalisé + features sélectionnées)
    dans data/processed/.
    """

    save_dir = get_processed_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename

    with open(path, "wb") as f:
        pickle.dump(df, f)

    print(f"[SAVE] Dataset prétraité sauvegardé dans : {path}")


# ---------------------------------------------------------
# 📥 Chargement du dataset prétraité
# ---------------------------------------------------------

def load_processed_dataset(
    filename: str = "processed_dataset.pkl",
) -> pd.DataFrame:
    """
    Charge le dataset prétraité depuis data/processed/.
    """

    path = get_processed_dir() / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Assure-toi d'avoir exécuté la pipeline de prétraitement."
        )

    with open(path, "rb") as f:
        df = pickle.load(f)

    print(f"[LOAD] Dataset prétraité chargé depuis : {path}")
    return df