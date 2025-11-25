# src/data/scaling.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------
# 🔧 Utilitaires
# ---------------------------------------------------------

def get_processed_dir() -> Path:
    """Retourne le dossier data/processed/."""
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def save_scaler(scaler, filename: str = "scaler.pkl"):
    """Sauvegarde du scaler (StandardScaler / MinMaxScaler)."""
    save_dir = get_processed_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename
    with open(path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[SAVE] Scaler sauvegardé dans : {path}")


# ---------------------------------------------------------
# 🚀 1. Fonction principale : normalisation des features
# ---------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scaler_type: str = "standard",
    fit: bool = True,
    scaler=None,
):
    """
    Normalise les features sélectionnées.
    
    Paramètres :
        - df : DataFrame à normaliser
        - feature_cols : colonnes à scaler
        - scaler_type : 'standard' ou 'minmax'
        - fit : True = fit + transform, False = transform only
        - scaler : scaler existant (pour le transform only)

    Retour :
        df_scaled, scaler
    """

    if scaler_type not in ("standard", "minmax"):
        raise ValueError("scaler_type doit être 'standard' ou 'minmax'.")

    # Choisir le scaler
    if scaler is None:
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    X = df[feature_cols].values

    if fit:
        print(f"[SCALING] Fitting scaler ({scaler_type}) sur {len(feature_cols)} features...")
        X_scaled = scaler.fit_transform(X)
        save_scaler(scaler)
    else:
        print(f"[SCALING] Transformation seule avec scaler existant...")
        X_scaled = scaler.transform(X)

    # Créer nouveau DataFrame
    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled

    print("[SCALING] Normalisation terminée.")

    return df_scaled, scaler