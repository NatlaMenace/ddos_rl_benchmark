# main.py

from __future__ import annotations

from pathlib import Path

from src.data.cicddos_loader import load_train_test_split
from src.data.export_processed import save_processed_dataset
from src.data.preprocessing import preprocess_cicddos
from src.data.feature_selection import combined_feature_selection, save_selected_features
from src.data.scaling import save_scaler, scale_features


def main():
    # 1) Chargement RAW : séparation naturelle TRAIN / TEST
    df_train_raw, df_test_raw = load_train_test_split()
    print("[INFO] RAW TRAIN shape :", df_train_raw.shape)
    print("[INFO] RAW TEST  shape :", df_test_raw.shape)

    # 2) Nettoyage (Phase 1) sur TRAIN et TEST
    print("\n[PREPROCESS] --- TRAIN ---")
    df_train_clean = preprocess_cicddos(df_train_raw)

    print("\n[PREPROCESS] --- TEST ---")
    df_test_clean = preprocess_cicddos(df_test_raw)

    # 3) Sélection de features (RF + MI) sur TRAIN uniquement
    print("\n[FS] --- Sélection de features sur TRAIN ---")
    top_features = combined_feature_selection(
        df_train_clean,
        label_col="Label",
        top_k=20,
    )
    print("[FS] Features sélectionnées :", top_features)

    # 4) Réduction aux features sélectionnées + Label
    df_train_reduced = df_train_clean[top_features + ["Label"]]
    df_test_reduced = df_test_clean[top_features + ["Label"]]

    # 5) Normalisation : fit sur TRAIN, transform sur TEST
    print("\n[SCALE] --- Standardisation TRAIN / TEST ---")
    df_train_scaled, scaler = scale_features(
        df_train_reduced,
        feature_cols=top_features,
        scaler_type="standard",
        fit=True,
    )

    df_test_scaled, _ = scale_features(
        df_test_reduced,
        feature_cols=top_features,
        scaler_type="standard",
        fit=False,
        scaler=scaler,
    )

    print("[SCALE] TRAIN final shape :", df_train_scaled.shape)
    print("[SCALE] TEST  final shape :", df_test_scaled.shape)

    # 6) Sauvegardes
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    save_processed_dataset(df_train_scaled, filename="processed_train_dataset.pkl")
    save_processed_dataset(df_test_scaled, filename="processed_test_dataset.pkl")
    save_scaler(scaler, filename="scaler.pkl")
    save_selected_features(top_features, filename="selected_features.json")

    print("\n[SAVE] Datasets et objets sauvegardés dans data/processed/")
    print("       - processed_train_dataset.pkl")
    print("       - processed_test_dataset.pkl")
    print("       - scaler.pkl")
    print("       - selected_features.json")


if __name__ == "__main__":
    main()