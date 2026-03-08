import pandas as pd

def select_features(df, dataset_name):
    """
    Sélectionne les features pertinentes selon le dataset.
    - Supprime les colonnes inutiles : Flow ID, Timestamp, adresses, etc.
    - Garde uniquement les colonnes utiles à l'entraînement ML.
    - Compatible CICIDS2017 et UNSW-NB15.

    Paramètres :
        df : DataFrame nettoyé (après US03)
        dataset_name : str ("CICIDS2017" ou "UNSW-NB15")

    Retour :
        df_selected : DataFrame avec seulement les colonnes pertinentes
    """

    print(f"\n🔍 Sélection des features pour {dataset_name}...")

    df_selected = df.copy()

    # =============================
    # 1) Colonnes inutiles communes
    # =============================
    COMMON_DROP = [
        "Flow ID", " Timestamp", "Source IP", "Destination IP", 
        "srcip", "dstip", "StartTime", "EndTime"
    ]

    df_selected = df_selected.drop(
        columns=[c for c in COMMON_DROP if c in df_selected.columns],
        errors="ignore"
    )

    # =============================
    # 2) CICIDS2017 — colonnes inutiles
    # =============================
    if dataset_name == "CICIDS2017":

        DROP_CICIDS = [
            " Fwd Header Length.1",  # doublon fréquent
            "Unnamed: 0",            # index inutile si existe
        ]

        df_selected = df_selected.drop(
            columns=[c for c in DROP_CICIDS if c in df_selected.columns],
            errors="ignore"
        )

        print("✔ Colonnes CICIDS inutiles supprimées.")

    # =============================
    # 3) UNSW-NB15 — colonnes inutiles
    # =============================
    if dataset_name == "UNSW-NB15":

        DROP_UNSW = [
            "id", "attack_cat", "label_str", "state_num"
        ]

        df_selected = df_selected.drop(
            columns=[c for c in DROP_UNSW if c in df_selected.columns],
            errors="ignore"
        )

        print("✔ Colonnes UNSW inutiles supprimées.")

    # =============================
    # 4) Final : vérification
    # =============================
    print(f"✔ Sélection terminée. Features restantes : {len(df_selected.columns)} colonnes.")
    return df_selected
