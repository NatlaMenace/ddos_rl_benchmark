# src/data/cicddos_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------- Chemins utilitaires ----------

def get_raw_data_dir() -> Path:
    """
    Retourne le dossier contenant les fichiers bruts CIC-DDoS2019.
    On part du principe que ce fichier est dans src/data/.
    """
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "cicddos2019"


def list_data_files(
    extensions: Sequence[str] = (".csv", ".parquet"),
    pattern: str = "*",
) -> List[Path]:
    """
    Liste les fichiers du dataset CIC-DDoS2019 dans data/raw/cicddos2019.

    - extensions : types de fichiers à considérer (.csv, .parquet, etc.)
    - pattern    : éventuellement restreindre (ex: 'UDP-*', '*training*', etc.)
    """
    data_dir = get_raw_data_dir()
    files: List[Path] = []

    for ext in extensions:
        files.extend(sorted(data_dir.glob(f"{pattern}{ext}")))

    return files

# ---------------------------------------------------------
# 🔀 Convenience functions for TRAIN / TEST splits
# ---------------------------------------------------------

def load_training_files(
    extensions=(".csv", ".parquet"),
):
    """Charge uniquement les fichiers dont le nom contient 'training'."""
    train_files = list_data_files(extensions=extensions, pattern="*training*")
    if not train_files:
        raise FileNotFoundError("Aucun fichier *training* trouvé dans data/raw/cicddos2019")
    return load_raw_files(files=train_files)


def load_testing_files(
    extensions=(".csv", ".parquet"),
):
    """Charge uniquement les fichiers dont le nom contient 'testing'."""
    test_files = list_data_files(extensions=extensions, pattern="*testing*")
    if not test_files:
        raise FileNotFoundError("Aucun fichier *testing* trouvé dans data/raw/cicddos2019")
    return load_raw_files(files=test_files)


def load_train_test_split():
    """
    Charge naturellement la séparation TRAIN / TEST
    en utilisant les noms de fichiers :
        *_training.parquet  → train
        *_testing.parquet   → test
    """
    print("[LOAD] Chargement TRAIN...")
    df_train = load_training_files()

    print("[LOAD] Chargement TEST...")
    df_test = load_testing_files()

    return df_train, df_test

# ---------- Chargement brut & concaténation ----------

def _read_single_file(path: Path) -> pd.DataFrame:
    """Lit un fichier CSV ou Parquet en fonction de son extension."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Extension non supportée pour {path}")

    # On conserve la provenance du flux (utile pour les découpages en épisodes)
    df["__source_file__"] = path.name
    return df


def load_raw_files(
    files: Optional[Iterable[Path]] = None,
    pattern: str = "*",
) -> pd.DataFrame:
    """
    Charge et concatène plusieurs fichiers CIC-DDoS2019.

    - files   : liste explicite de Path. Si None, on utilise `pattern`.
    - pattern : motif de filtrage si `files` n'est pas fourni.

    Retour : DataFrame concaténé (sans filtrage de classes pour l’instant).
    """
    if files is None:
        files = list_data_files(pattern=pattern)

    files = list(files)
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {get_raw_data_dir()} avec le pattern '{pattern}'"
        )

    dfs = []
    for path in files:
        print(f"[LOAD] {path}")
        dfs.append(_read_single_file(path))

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Dataset concaténé : {len(df):,} lignes")
    return df


# ---------- Sélection de sous-ensemble (classes & taille) ----------

def select_classes_and_limit(
    df: pd.DataFrame,
    label_col: str = "Label",
    classes: Optional[Sequence[str]] = None,
    max_rows_per_class: Optional[int] = None,
    random_state: int = 42,
    keep_order: bool = False,
) -> pd.DataFrame:
    """
    Sélectionne un sous-ensemble du dataset :

    - `classes` : liste de labels à garder (None -> toutes les classes)
    - `max_rows_per_class` : limite de lignes par classe (échantillonnage équilibré)
    - `keep_order` : si False, on mélange les lignes après sous-échantillonnage.
                     si True, on conserve l’ordre (pour l’aspect temporel).

    Retour : DataFrame filtré.
    """
    if label_col not in df.columns:
        raise KeyError(
            f"Colonne de label '{label_col}' introuvable. "
            f"Colonnes disponibles : {list(df.columns)[:10]}..."
        )

    work_df = df

    # 1) Filtrage des classes si demandé
    if classes is not None:
        classes_lower = [c.lower() for c in classes]
        work_df = work_df[work_df[label_col].str.lower().isin(classes_lower)]
        print(
            f"[FILTER] Classes gardées = {classes} -> {len(work_df):,} lignes après filtrage"
        )

    # 2) Limitation du nombre de lignes par classe
    if max_rows_per_class is not None:
        sampled_parts = []
        for label, group in work_df.groupby(label_col):
            if len(group) > max_rows_per_class:
                group = group.sample(
                    n=max_rows_per_class, random_state=random_state
                )
            sampled_parts.append(group)
        work_df = pd.concat(sampled_parts, ignore_index=True)
        print(
            f"[FILTER] Limite de {max_rows_per_class} lignes par classe -> {len(work_df):,} lignes"
        )

    # 3) Mélange optionnel
    if not keep_order:
        work_df = work_df.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )

    return work_df


# ---------- Gestion de la structure temporelle ----------

def sort_by_time(
    df: pd.DataFrame,
    time_col_candidates: Sequence[str] = ("Timestamp", "Flow Start Timestamp"),
) -> pd.DataFrame:
    """
    Trie le DataFrame par timestamp si une colonne temporelle pertinente est trouvée.

    On essaie quelques noms de colonnes typiques du CIC-DDoS2019 :
    - 'Timestamp'
    - 'Flow Start Timestamp'
    (tu pourras adapter cette liste après inspection réelle des colonnes)

    Si aucune colonne n'est trouvée, le df est retourné tel quel.
    """
    for col in time_col_candidates:
        if col in df.columns:
            print(f"[TIME] Tri par la colonne temporelle '{col}'")
            return df.sort_values(col).reset_index(drop=True)

    print("[TIME] Aucune colonne temporelle trouvée, ordre inchangé.")
    return df


# ---------- Pipeline haut niveau pratique ----------

def load_cicddos_subset(
    pattern: str = "*",
    label_col: str = "Label",
    classes: Optional[Sequence[str]] = None,
    max_rows_per_class: Optional[int] = None,
    keep_temporal_order: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Pipeline complet pour les premiers tests :

    1) Charge et concatène les fichiers correspondant au `pattern`.
    2) Trie par temps si possible (structure temporelle).
    3) Filtre éventuellement les classes et limite le nombre de lignes par classe.

    Exemple d’usage :
        df = load_cicddos_subset(
            pattern="UDP-*",
            classes=["BENIGN", "UDP"],
            max_rows_per_class=50_000,
        )
    """
    df = load_raw_files(pattern=pattern)

    print("Test %s",df[label_col].value_counts)

    if keep_temporal_order:
        df = sort_by_time(df)

    df = select_classes_and_limit(
        df,
        label_col=label_col,
        classes=classes,
        max_rows_per_class=max_rows_per_class,
        random_state=random_state,
        keep_order=keep_temporal_order,
    )

    print(
        f"[DONE] Sous-ensemble final : {len(df):,} lignes, "
        f"{df[label_col].nunique()} classes"
    )
    return df