import pickle
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 🏋️‍♂️ Entraînement du modèle Random Forest
# ============================================================
def train_rf(X_train, y_train, seed=42, n_estimators=100):
    """
    Entraîne un modèle Random Forest baseline.
    
    - Pas de tuning, modèle simple et reproductible.
    - Aucune transformation ou preprocessing ici (déjà fait en amont).

    Retour :
        model : modèle entraîné
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1  # utilise tous les CPU
    )

    print("🔥 Entraînement du Random Forest...")
    model.fit(X_train, y_train.values.ravel())
    print("✔️ Modèle RF entraîné avec succès.")

    return model


# ============================================================
# 💾 Sauvegarde du modèle
# ============================================================
def save_model(model, path):
    """
    Sauvegarde le modèle Random Forest au format pickle (.pkl).
    """

    import os
    
    # --- Nouveauté importante ---
    # Création automatique des dossiers si inexistants
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # --------------------------------

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Modèle sauvegardé sous : {path}")


# ============================================================
# 📥 Chargement du modèle
# ============================================================
def load_model(path):
    """
    Charge un modèle sauvegardé au format pickle (.pkl).
    """

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(f"📁 Modèle chargé depuis : {path}")
    return model


# ============================================================
# 📥 Entrainement du modèle sur un seul decision Tree avec profondeur limité
# ============================================================
from sklearn.tree import DecisionTreeClassifier
from time import sleep
from tqdm import tqdm
def train_dt_baseline(X_train, y_train, max_depth=4, seed=42):
    """
    Entraîne un Decision Tree simple avec affichage clair :
    - message début
    - barre de progression factice pour feedback
    - message fin
    """

    print(f"\n🌳 Initialisation du Decision Tree (max_depth={max_depth})...")

    

    # Création du modèle
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)

    print("\n🔥 Entraînement du modèle...")
  

    # Entraînement réel
    model.fit(X_train, y_train.values.ravel())

    print("\n✔️ Modèle Decision Tree entraîné avec succès !\n")

    return model