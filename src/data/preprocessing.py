

"""
Prétraitement du dataset CIC-DDoS2019.

Ce module centralise toutes les étapes de préparation des données :
- chargement et fusion des fichiers .parquet bruts ;
- nettoyage basique (NA, colonnes vides) ;
- sélection des features et de la cible ;
- normalisation / standardisation ;
- découpage train / test ;
- sauvegarde dans data/processed/.

⚠️ IMPORTANT :
Les noms de colonnes du dataset peuvent varier légèrement d'une source à l'autre.
Si une KeyError apparaît lors de la sélection des features, adapter la liste
des colonnes dans la configuration ci-dessous ou passer vos propres listes
à `select_features(...)` ou `build_train_test_sets(...)`.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Répertoires par défaut
RAW_DIR = Path("data/raw/cicddos2019")
PROCESSED_DIR = Path("data/processed")

# Cible par défaut (à adapter si besoin)
DEFAULT_TARGET_COLUMN = "Label"

# Liste indicative de features courantes dans CIC-DDoS2019.
# Cette liste est volontairement "souple" : on prendra seulement
# l'intersection avec les colonnes réellement présentes dans le DataFrame.
CICDDOS2019_FEATURE_CANDIDATES: List[str] = [
    # Durée et volumes
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    # Inter-arrival times
    "Fwd IAT Tot",
    "Bwd IAT Tot",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    # Flags
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    # Protocole
    "Protocol",
]


def load_raw_data(raw_dir: Path | str = RAW_DIR, pattern: str = "*.parquet") -> pd.DataFrame:
    """
    Charge et fusionne tous les fichiers .parquet du dossier brut.

    Parameters
    ----------
    raw_dir : Path or str
        Dossier contenant les fichiers bruts CIC-DDoS2019.
    pattern : str
        Motif de fichiers à charger (par défaut '*.parquet').

    Returns
    -------
    pd.DataFrame
        DataFrame fusionné contenant toutes les lignes.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Le dossier brut n'existe pas : {raw_path.resolve()}")

    files = sorted(raw_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier '{pattern}' trouvé dans {raw_path.resolve()}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        print(f"[LOAD] Lecture de {f.name}")
        df_part = pd.read_parquet(f)
        df_part["__source_file__"] = f.name  # utile pour débogage / traçabilité
        dfs.append(df_part)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total lignes fusionnées : {len(full_df)}")
    print(f"[INFO] Nombre de colonnes : {full_df.shape[1]}")
    return full_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage simple du DataFrame :
    - suppression des colonnes entièrement vides ;
    - remplacement des valeurs manquantes par 0.

    À adapter si nécessaire pour un nettoyage plus fin.
    """
    # Supprimer les colonnes entièrement vides
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after_cols = df.shape[1]
    if after_cols < before_cols:
        print(f"[CLEAN] Colonnes entièrement vides supprimées : {before_cols - after_cols}")

    # Remplacer les NA restants
    na_count = df.isna().sum().sum()
    if na_count > 0:
        print(f"[CLEAN] Remplacement de {na_count} valeurs manquantes par 0")
        df = df.fillna(0)

    return df


def infer_default_feature_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    """
    Devine une liste de features raisonnable :
    - si des colonnes candidates CICDDOS2019 existent, prend leur intersection ;
    - sinon, prend toutes les colonnes numériques sauf la cible.
    """
    existing_candidates = [c for c in CICDDOS2019_FEATURE_CANDIDATES if c in df.columns]

    if existing_candidates:
        print(f"[FEATURES] Utilisation des features candidates présentes : {len(existing_candidates)}")
        return existing_candidates

    # Fallback : toutes les colonnes numériques sauf la cible
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    print(f"[FEATURES] Aucune feature candidate trouvée, utilisation de toutes les colonnes numériques ({len(numeric_cols)})")
    return numeric_cols


def select_features(
    df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sélectionne X (features) et y (cible) à partir du DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Données brutes nettoyées.
    feature_columns : iterable of str, optional
        Liste des colonnes à utiliser comme features. Si None,
        une heuristique tente de les deviner.
    target_column : str
        Nom de la colonne cible.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if target_column not in df.columns:
        raise KeyError(
            f"La colonne cible '{target_column}' est introuvable dans le DataFrame. "
            f"Colonnes disponibles : {list(df.columns)[:20]}..."
        )

    if feature_columns is None:
        feature_columns = infer_default_feature_columns(df, target_column)
    else:
        feature_columns = list(feature_columns)

    # Ne garder que les colonnes qui existent réellement
    feature_columns = [c for c in feature_columns if c in df.columns]
    if not feature_columns:
        raise ValueError(
            "Aucune colonne de features valide trouvée. "
            "Vérifier les noms de colonnes ou adapter CICDDOS2019_FEATURE_CANDIDATES."
        )

    print(f"[FEATURES] Nombre de features utilisées : {len(feature_columns)}")

    X = df[feature_columns]
    y = df[target_column].astype("category").cat.codes

    return X, y


def build_train_test_sets(
    raw_dir: Path | str = RAW_DIR,
    feature_columns: Optional[Iterable[str]] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Pipeline complet :
    - chargement des données brutes ;
    - nettoyage ;
    - sélection des features et de la cible ;
    - normalisation StandardScaler ;
    - split train / test.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    df_raw = load_raw_data(raw_dir=raw_dir)
    df_clean = clean_dataframe(df_raw)
    X_df, y_series = select_features(df_clean, feature_columns=feature_columns, target_column=target_column)

    # Conversion en numpy
    X = X_df.to_numpy()
    y = y_series.to_numpy()

    # Split train / test
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    print(f"[SPLIT] X_train : {X_train.shape}, X_test : {X_test.shape}")

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    out_dir: Path | str = PROCESSED_DIR,
) -> None:
    """
    Sauvegarde les jeux de données prétraités et le scaler dans data/processed/.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "X_train.npy", X_train)
    np.save(out_path / "X_test.npy", X_test)
    np.save(out_path / "y_train.npy", y_train)
    np.save(out_path / "y_test.npy", y_test)
    joblib.dump(scaler, out_path / "scaler.pkl")

    print(f"[SAVE] Données prétraitées sauvegardées dans {out_path.resolve()}")


def run_full_preprocessing_pipeline() -> None:
    """
    Fonction utilitaire pour lancer le pipeline complet depuis la ligne de commande.

    Exemple :
    ---------
    >>> python -m src.data.preprocessing
    """
    print("[PIPELINE] Démarrage du prétraitement complet CIC-DDoS2019...")
    X_train, X_test, y_train, y_test, scaler = build_train_test_sets()
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    print("[PIPELINE] Prétraitement terminé avec succès ✅")


if __name__ == "__main__":
    run_full_preprocessing_pipeline()