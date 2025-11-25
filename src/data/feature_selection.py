# src/data/feature_selection.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def _get_numeric_feature_cols(df: pd.DataFrame, label_col: str) -> List[str]:
    """
    Retourne la liste des colonnes numériques utilisables comme features
    (on exclut la colonne de label).
    """
    X = df.drop(columns=[label_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

# ---------------------------------------------------------
# 🔧 1) Utilitaires pour sauvegarde / chemins
# ---------------------------------------------------------

def get_processed_dir() -> Path:
    """Retourne le dossier data/processed/."""
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def save_selected_features(features: Sequence[str], filename: str = "selected_features.json"):
    """Sauvegarde la liste des features dans data/processed/."""
    save_dir = get_processed_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename
    with open(path, "w") as f:
        json.dump(features, f, indent=4)

    print(f"[SAVE] Features sauvegardées dans : {path}")


# ---------------------------------------------------------
# 🔍 2) Sélection via RandomForestClassifier
# ---------------------------------------------------------

def select_features_random_forest(
    df: pd.DataFrame,
    label_col: str,
    top_k: Optional[int] = 20,
    n_estimators: int = 200,
    random_state: int = 42,
) -> List[Tuple[str, float]]:
    """
    Sélection de features basée sur l'importance d'un RandomForestClassifier.
    Retour : liste triée (feature, importance)
    """

    print("[FS-RF] Sélection de features via RandomForestClassifier...")

    # 🔹 On ne garde que les colonnes numériques comme features
    feature_cols = _get_numeric_feature_cols(df, label_col)
    X = df[feature_cols]
    y = df[label_col]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_importances = list(zip(feature_cols, importances))

    feature_importances.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        feature_importances = feature_importances[:top_k]

    print(f"[FS-RF] Top {top_k} features :")
    for feat, score in feature_importances[:10]:
        print(f"   - {feat} : {score:.4f}")

    return feature_importances


# ---------------------------------------------------------
# 🔍 3) Sélection via Mutual Information
# ---------------------------------------------------------

def select_features_mutual_info(
    df: pd.DataFrame,
    label_col: str,
    top_k: Optional[int] = 20,
    random_state: int = 42,
) -> List[Tuple[str, float]]:
    """
    Sélection de features basée sur la Mutual Information.
    Retour : liste triée (feature, score)
    """

    print("[FS-MI] Sélection de features via Mutual Information...")

    feature_cols = _get_numeric_feature_cols(df, label_col)
    X = df[feature_cols]
    y = df[label_col]

    mi_scores = mutual_info_classif(
        X, y, random_state=random_state, n_neighbors=3
    )

    feature_mi = list(zip(feature_cols, mi_scores))
    feature_mi.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        feature_mi = feature_mi[:top_k]

    print(f"[FS-MI] Top {top_k} features :")
    for feat, score in feature_mi[:10]:
        print(f"   - {feat} : {score:.4f}")

    return feature_mi


# ---------------------------------------------------------
# 🔗 4) Pipeline final combiné (RF + MI)
# ---------------------------------------------------------

def combined_feature_selection(
    df: pd.DataFrame,
    label_col: str = "Label",
    top_k: int = 20,
    weight_rf: float = 0.6,
    weight_mi: float = 0.4,
) -> List[str]:
    """
    Combinaison de deux méthodes :
    - RandomForest importance (supervisé, robuste)
    - Mutual Information (moins biaisé)

    La combinaison stabilise le choix des features.
    """

    print("\n[FS] --- DÉBUT DE LA SÉLECTION DE FEATURES ---")

    # 🔹 On restreint déjà aux features numériques
    feature_cols = _get_numeric_feature_cols(df, label_col)
    df_num = df[feature_cols + [label_col]]

    rf_importances = select_features_random_forest(
        df_num, label_col=label_col, top_k=None
    )
    mi_importances = select_features_mutual_info(
        df_num, label_col=label_col, top_k=None
    )

    rf_dict = dict(rf_importances)
    mi_dict = dict(mi_importances)

    combined_scores = {}

    for feature in feature_cols:
        combined_scores[feature] = (
            rf_dict.get(feature, 0.0) * weight_rf +
            mi_dict.get(feature, 0.0) * weight_mi
        )

    sorted_features = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = [feat for feat, score in sorted_features[:top_k]]

    print("\n[FS] --- FEATURES SÉLECTIONNÉES ---")
    for f in top_features:
        print(f"   - {f}")

    save_selected_features(top_features)

    return top_features