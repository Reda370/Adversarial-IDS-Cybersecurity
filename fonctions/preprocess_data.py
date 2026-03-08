import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1) EXTRACTION DU LABEL
# ============================================================
def extract_label(df, dataset_name):
    """
    Extrait X et y selon le dataset.
    CICIDS2017 -> label = ' Label'
    UNSW-NB15 -> label = 'label'
    """
    print("🔎 Extraction du label...")

    if dataset_name == "CICIDS2017":
        y = df["Label"]
        X = df.drop(columns=["Label"])
    else:  # UNSW
        y = df["label"]
        X = df.drop(columns=["label"])

    return X, y



# ============================================================
# 2) ENCODAGE DES COLONNES CATEGORIELLES (UNSW SEULEMENT)
# ============================================================
def encode_categorical(X, dataset_name):
    """
    One-Hot Encoding pour UNSW uniquement.
    CICIDS -> retourne X inchangé.
    """
    if dataset_name != "UNSW-NB15":
        print("ℹ Aucun encodage nécessaire pour CICIDS.")
        return X

    print("🔧 Encodage One-Hot pour UNSW...")

    cat_cols = ["proto", "service", "state"]
    cat_cols = [c for c in cat_cols if c in X.columns]

    if len(cat_cols) == 0:
        print("ℹ Aucune colonne catégorielle trouvée.")
        return X

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Fusion
    X = X.drop(columns=cat_cols)
    X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

    return X



# ============================================================
# 3) NORMALISATION DES FEATURES (OPTIONNEL)
# ============================================================
def normalize_features(X):
    """
    Normalisation StandardScaler (utile pour MLP).
    Random Forest n’en a pas besoin.
    """
    print("📏 Normalisation des features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return X



# ============================================================
# 4) PETITE PIPELINE (OPTIONNELLE)
# ============================================================
def preprocess_pipeline(df, dataset_name, normalize=False):
    """
    Option : petite fonction pour enchainer les étapes.
    (Tu n'es PAS obligé de l'utiliser si tu veux tout faire dans le main.)
    """
    X, y = extract_label(df, dataset_name)
    X = encode_categorical(X, dataset_name)

    if normalize:
        X = normalize_features(X)

    print("✔ Préprocessing terminé.")
    return X, y
