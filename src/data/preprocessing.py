# src/data/preprocessing.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence


# --------------------
# 1) Détection & suppression de colonnes inutiles
# --------------------

def drop_useless_columns(
    df: pd.DataFrame,
    drop_cols: Optional[Sequence[str]] = None,
    auto_detect_timestamps: bool = True,
) -> pd.DataFrame:
    """
    Supprime les colonnes inutiles :
        - IDs, index, metadata
        - timestamps si non utilisés pour l'env RL
        - colonnes vides ou constantes
    """

    work_df = df.copy()
    
    # Colonnes à retirer manuellement
    if drop_cols is not None:
        for col in drop_cols:
            if col in work_df.columns:
                print(f"[CLEAN] Suppression colonne : {col}")
                work_df = work_df.drop(columns=[col])

    meta_cols = ["Flow ID", "Src IP", "Dst IP", "__source_file__"]
    for col in meta_cols:
        if col in work_df.columns:
            print(f"[CLEAN] Suppression colonne meta : {col}")
            work_df = work_df.drop(columns=[col])

    # Détection automatique de colonnes timestamp si demandé
    if auto_detect_timestamps:
        ts_candidates = [
            "Timestamp",
            "Flow Start Timestamp",
            "Flow End Timestamp",
            "Date",
            "Time",
        ]
        for col in ts_candidates:
            if col in work_df.columns:
                print(f"[CLEAN] Suppression colonne timestamp inutilisée : {col}")
                work_df = work_df.drop(columns=[col])

    # Retirer colonnes constantes
    const_cols = [c for c in work_df.columns if work_df[c].nunique() <= 1]
    if const_cols:
        print(f"[CLEAN] Colonnes constantes supprimées : {const_cols}")
        work_df = work_df.drop(columns=const_cols)

    # Colonnes entièrement NA
    na_cols = [c for c in work_df.columns if work_df[c].isna().all()]
    if na_cols:
        print(f"[CLEAN] Colonnes entièrement NA supprimées : {na_cols}")
        work_df = work_df.drop(columns=na_cols)

    return work_df


# --------------------
# 2) Nettoyage des NaN et valeurs infinies
# --------------------

def clean_missing_and_infinite(
    df: pd.DataFrame,
    numeric_imputation: Optional[str] = "median",
    fill_value: Optional[float] = 0.0,
) -> pd.DataFrame:
    """
    Nettoie toutes les valeurs NaN ou infinies.

    - numeric_imputation = "median" | "mean" | "zero" | None
    - fill_value : si numeric_imputation="zero", la valeur utilisée

    Steps :
        - remplacer inf / -inf par NaN
        - imputer NaN selon la stratégie
    """

    work_df = df.copy()

    # 1) Remplacer inf par NaN
    work_df = work_df.replace([np.inf, -np.inf], np.nan)

    # 2) Imputation
    if numeric_imputation is None:
        print("[CLEAN] Aucune imputation appliquée.")
        return work_df

    numeric_cols = work_df.select_dtypes(include=[np.number]).columns

    print(f"[CLEAN] Imputation des NaN sur {len(numeric_cols)} colonnes numériques via : {numeric_imputation}")

    if numeric_imputation == "median":
        imputer = work_df[numeric_cols].median()
    elif numeric_imputation == "mean":
        imputer = work_df[numeric_cols].mean()
    elif numeric_imputation == "zero":
        imputer = {col: fill_value for col in numeric_cols}
    else:
        raise ValueError(f"Méthode d'imputation inconnue : {numeric_imputation}")

    work_df[numeric_cols] = work_df[numeric_cols].fillna(imputer)

    # Vérification finale
    remaining_na = work_df.isna().sum().sum()
    if remaining_na > 0:
        print(f"[WARN] {remaining_na} valeurs NaN restantes (colonnes non numériques ou autres cas).")
    else:
        print("[CLEAN] Plus aucune valeur NaN dans le DataFrame.")

    return work_df


# --------------------
# 3) Pipeline complet
# --------------------

def preprocess_cicddos(
    df: pd.DataFrame,
    drop_cols: Optional[Sequence[str]] = None,
    imputation: str = "median",
    remove_timestamps: bool = True,
) -> pd.DataFrame:
    """
    Pipeline de nettoyage complet :
    1. Suppression de colonnes inutiles
    2. Nettoyage NaN / inf
    """

    print("[PREPROCESS] --- Début du nettoyage du dataset ---")

    df = drop_useless_columns(
        df,
        drop_cols=drop_cols,
        auto_detect_timestamps=remove_timestamps,
    )

    df = clean_missing_and_infinite(
        df,
        numeric_imputation=imputation,
    )

    print("[PREPROCESS] --- Nettoyage terminé ---")
    return df