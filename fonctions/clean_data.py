import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Détection automatique du dataset
# ---------------------------------------------------------
def detect_dataset(df):
    """
    Détection robuste du dataset CICIDS2017 ou UNSW-NB15.
    Compatible avec les colonnes exactes de CICIDS2017 (y compris ' Label').
    """

    cols = df.columns.str.lower()

    # ---- CICIDS2017 ----
    cicids_cols = [
        "flow bytes/s",
        "flow packets/s",
        "flow duration",
        "total fwd packets",
        "total backward packets",
        " label"  # attention l'espace avant !
    ]

    if any(col in cols for col in cicids_cols):
        return "CICIDS2017"

    # ---- UNSW-NB15 ----
    unsw_cols = [
        "attack_cat",
        "proto",
        "sport",
        "dsport",
        "ct_state_ttl"
    ]

    if any(col in cols for col in unsw_cols):
        return "UNSW-NB15"

    return "UNKNOWN"



# ---------------------------------------------------------
# Nettoyage complet du dataset (US03)
# ---------------------------------------------------------
def clean_dataset(df):
    """
    Nettoyage US03 :
    - Fonction compatible CICIDS2017 & UNSW-NB15
    - Analyse des NaN
    - Suppression NaN + suppression inf / -inf
    - Nettoyage des outliers selon les caractéristiques du dataset

    Paramètre :
        df : DataFrame brut du dataset

    Retour :
        df_clean : DataFrame nettoyé
    """

    print("\n=====================================")
    print("🧹 US03 – Nettoyage des données")
    print("=====================================\n")

    # --------------------------------------------
    # 0) Détection du dataset
    # --------------------------------------------
    dataset_type = detect_dataset(df)
    print(f"📌 Dataset détecté : {dataset_type}")

    # --------------------------------------------
    # 1) Rapport des NaN
    # --------------------------------------------
    print("\n📊 Pourcentage de NaN par colonne (affichage > 0%) :")
    nan_report = df.isnull().mean() * 100
    print(nan_report[nan_report > 0].sort_values(ascending=False))

    # --------------------------------------------
    # 2) Suppression des NaN + inf
    # --------------------------------------------
    print("\n🧽 Suppression des NaN et valeurs infinies...")
    before_nan = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)  # Convertir inf → NaN
    df = df.dropna()

    after_nan = len(df)
    print(f"✔ Lignes supprimées (NaN + inf): {before_nan - after_nan}")

    # --------------------------------------------
    # 3) Traitement des outliers
    # --------------------------------------------
    print("\n📌 Suppression des outliers extrêmes...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Seuils adaptés selon le dataset
    if dataset_type == "CICIDS2017":
        percentile = 0.995   # plus d'outliers
        print("  → CICIDS2017 : seuil 99.5%")
    elif dataset_type == "UNSW-NB15":
        percentile = 0.999   # dataset plus stable
        print("  → UNSW-NB15 : seuil 99.9%")
    else:
        percentile = 0.995
        print("  → Dataset inconnu : seuil par défaut 99.5%")

    # Calcul des bornes
    upper_bounds = df[numeric_cols].quantile(percentile)

    before_outliers = len(df)

    for col in numeric_cols:
        df = df[df[col] <= upper_bounds[col]]

    after_outliers = len(df)

    print(f"✔ Lignes supprimées (outliers) : {before_outliers - after_outliers}")

    print("\n🎉 Nettoyage terminé !")
    print(f"📏 Nombre final de lignes : {len(df)}\n")

    return df
