# ============================================================
#  MLP_baseline.py
#  Modèle MLP baseline + Entraînement compatible CICIDS / UNSW
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================
# ⭐ MODELE MLP BASELINE (simple et robuste)
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    # ========================================================
    # ⭐ MÉTHODE .predict() POUR COMPATIBILITÉ SKLEARN
    # ========================================================
    def predict(self, X):
        """
        Prédit les classes comme un modèle sklearn.
        Utilisé par feature_impact.py.

        Supporte automatiquement :
            - sortie 2 neurones (softmax → argmax)
            - sortie 1 neurone (sigmoid → seuil 0.5)
        """
        self.eval()

        device = next(self.parameters()).device

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.forward(X_tensor)

            # Cas 2 neurones : classification softmax → argmax
            if logits.shape[1] == 2:
                preds = logits.argmax(dim=1)

            # Cas 1 neurone : sigmoid → seuil 0.5
            else:
                preds = (logits >= 0.5).int()

        return preds.cpu().numpy().ravel()


# ============================================================
# ⭐ FONCTION : Entraînement + Évaluation + Sauvegarde
# ============================================================

def train_and_save_mlp(
        X_train, y_train,
        X_test, y_test,
        save_path="mlp_model.pt",
        epochs=12,
        batch_size=256,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    Fonction complète d'entraînement MLP baseline.

    ✦ Compatible CICIDS2017 et UNSW-NB15
    ✦ Filtre automatiquement les colonnes numériques
    ✦ Convertit en float32 / int64 (PyTorch ready)
    ✦ Sauvegarde le modèle dans save_path
    """

    print("🚀 Entraînement MLP baseline...")

    # --------------------------------------------------------
    # 0. S'assurer que c'est bien un DataFrame / Series
    # --------------------------------------------------------
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    if not isinstance(y_train, (pd.Series, pd.DataFrame)):
        y_train = pd.Series(y_train)

    if not isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = pd.Series(y_test)

    # --------------------------------------------------------
    # 1. Garder UNIQUEMENT les colonnes numériques
    # --------------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("❌ Erreur : aucune colonne numérique détectée. "
                         "Vérifie ton preprocessing / encodage.")

    # Conversion → numpy float32
    X_train_np = X_train[numeric_cols].to_numpy(dtype=np.float32)
    X_test_np  = X_test[numeric_cols].to_numpy(dtype=np.float32)

    # Labels → int64
    y_train_np = pd.to_numeric(y_train, errors="raise").to_numpy(dtype=np.int64)
    y_test_np  = pd.to_numeric(y_test, errors="raise").to_numpy(dtype=np.int64)

    # --------------------------------------------------------
    # 2. Conversion en tenseurs PyTorch
    # --------------------------------------------------------
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.long)

    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_np, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --------------------------------------------------------
    # 3. Initialisation du modèle
    # --------------------------------------------------------
    input_dim = X_train_t.shape[1]
    model = MLPBaseline(input_dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --------------------------------------------------------
    # 4. Boucle d'entraînement
    # --------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📘 Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    # --------------------------------------------------------
    # 5. Évaluation
    # --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device)).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test_np, preds)
    print(f"\n🎯 Accuracy test : {acc:.4f}")

    # --------------------------------------------------------
    # 6. Sauvegarde du modèle
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"💾 Modèle sauvegardé dans : {save_path}")

    return model, preds




def clean_cicids_for_MLP(df, test_size=0.2, seed=42):
    """
    Nettoie le dataset CICIDS2017 et retourne :
    X_train, X_test, y_train, y_test

    Étapes :
    - Harmoniser nom du label (" Label" → "Label")
    - Strip() de la colonne Label
    - Binarisation : BENIGN = 0 / ATTACK = 1
    - Suppression colonnes inutiles
    - Train/test split
    """

    # === 1) Corriger nom colonne label ===
    if " Label" in df.columns:
        df["Label"] = df[" Label"]
        df = df.drop(columns=[" Label"])

    # === 2) Nettoyage du label ===
    df["Label"] = df["Label"].astype(str).str.strip()

    # === 3) Binarisation ===
    df["binary_label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # === 4) Séparation X / y ===
    X = df.drop(columns=["Label", "binary_label"], errors="ignore")
    y = df["binary_label"]

    # === 5) Split train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    print("✨ CICIDS nettoyé et split effectué.")
    print(f"  → Train : {len(X_train)} lignes")
    print(f"  → Test  : {len(X_test)} lignes")
    print(f"  → Attacks ratio (train) : {y_train.mean():.4f}")

    return X_train, X_test, y_train, y_test


# ============================================================
#  prepare_cicids.py
#  Pipeline complet de préparation CICIDS pour MLP
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_cicids_for_mlp(X_train, X_val, X_test,
                           y_train, y_val, y_test,
                           benign_names=["BENIGN"]):
    """
    Prépare les données CICIDS pour entraîner un MLP PyTorch.
    
    Étapes :
    - Conversion y DataFrame → Series
    - Binarisation du label (0 = BENIGN, 1 = Attack)
    - Suppression colonne 'Label' dans X
    - Conserver uniquement les colonnes numériques
    - Normalisation (fit sur train, transform sur val et test)

    Retourne :
        X_train_norm, X_val_norm, X_test_norm,
        y_train_bin, y_val_bin, y_test_bin
    """

    # =========================================
    # 1) Convertir Y en Series propre
    # =========================================
    y_train = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
    y_val   = y_val.iloc[:, 0]   if isinstance(y_val, pd.DataFrame) else y_val
    y_test  = y_test.iloc[:, 0]  if isinstance(y_test, pd.DataFrame) else y_test

    # =========================================
    # 2) Binarisation 0 / 1
    # =========================================
    y_train = y_train.astype(str).str.strip().apply(lambda x: 0 if x in benign_names else 1)
    y_val   = y_val.astype(str).str.strip().apply(lambda x: 0 if x in benign_names else 1)
    y_test  = y_test.astype(str).str.strip().apply(lambda x: 0 if x in benign_names else 1)

    # =========================================
    # 3) Nettoyage des features X
    # =========================================
    # a) Supprimer la colonne 'Label' si elle existe
    X_train = X_train.drop(columns=["Label"], errors="ignore")
    X_val   = X_val.drop(columns=["Label"], errors="ignore")
    X_test  = X_test.drop(columns=["Label"], errors="ignore")

    # b) Garder uniquement les colonnes numériques
    num_cols = X_train.select_dtypes(include=["number"]).columns

    X_train = X_train[num_cols]
    X_val   = X_val[num_cols]
    X_test  = X_test[num_cols]

    # =========================================
    # 4) Normalisation (fit sur TRAIN uniquement)
    # =========================================
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_val_norm   = scaler.transform(X_val)
    X_test_norm  = scaler.transform(X_test)

    return (
        X_train_norm, X_val_norm, X_test_norm,
        y_train, y_val, y_test,
        num_cols,
        scaler     # ← IMPORTANT
    )



# ============================================================
# prepare_balanced.py
# Préparation des données équilibrées pour MLP
# ============================================================

import pandas as pd
import numpy as np


def prepare_balanced_for_mlp(X_train_bal, y_train_bal, num_cols, scaler, benign_names=["BENIGN"]):
    """
    Prépare les données oversampled/undersampled pour l'entraînement MLP.

    Paramètres
    ----------
    X_train_bal : DataFrame
        Données d'entraînement rééquilibrées (X_train après oversampling)
    y_train_bal : DataFrame ou Series
        Labels rééquilibrés
    num_cols : liste
        Colonnes numériques utilisées dans le modèle baseline
    scaler : StandardScaler (déjà fit sur X_train original)
        Scaler appris sur les données TRAIN ORIGINALES
    benign_names : list
        Liste des labels correspondant au trafic normal (BENIGN)

    Retourne
    --------
    X_train_bal_norm : ndarray
        Données équilibrées normalisées (même norme que baseline)
    y_train_bal_bin : Series
        Labels binarisés (0 = Benign, 1 = Attack)
    """

    # --- 1) Convertir en Series si DataFrame ---
    if isinstance(y_train_bal, pd.DataFrame):
        y_train_bal = y_train_bal.iloc[:, 0]

    # --- 2) Binarisation du label ---
    y_train_bal_bin = y_train_bal.astype(str).str.strip().apply(
        lambda x: 0 if x in benign_names else 1
    )

    # --- 3) Garder uniquement les colonnes numériques déjà utilisées ---
    X_train_bal = X_train_bal[num_cols]

    # --- 4) Normalisation avec le scaler d'origine ---
    X_train_bal_norm = scaler.transform(X_train_bal)

    return X_train_bal_norm, y_train_bal_bin



def prepare_unsw_for_mlp(X_train_u, X_val_u, X_test_u,
                         y_train_u, y_val_u, y_test_u):
    """
    Prépare les données UNSW-NB15 pour entraîner un MLP PyTorch :
    
    Étapes :
    - Convertir y DataFrame -> Series
    - Label UNSW déjà binaire (0/1) -> juste conversion propre
    - Supprimer les colonnes inutiles : label, attack_cat
    - Garder uniquement les colonnes numériques
    - Normalisation (fit sur train, transform sur val + test)

    Retourne :
    - X_train_u_mlp, X_val_u_mlp, X_test_u_mlp : arrays normalisés
    - y_train_u, y_val_u, y_test_u : labels propres
    - num_cols_u : liste des features utilisées
    - scaler_u : scaler (pour appliquer à oversampling/SMOTE plus tard)
    """

    # ============================
    # 1) Convertir Y en Series
    # ============================
    y_train_u = y_train_u.squeeze()
    y_val_u   = y_val_u.squeeze()
    y_test_u  = y_test_u.squeeze()

    # (UNSW est déjà binaire donc aucune binarisation needed)

    # ============================
    # 2) Supprimer colonnes inutiles des X
    # ============================
    cols_to_drop = ["label", "Label", "attack_cat"]
    X_train_u = X_train_u.drop(columns=cols_to_drop, errors="ignore")
    X_val_u   = X_val_u.drop(columns=cols_to_drop, errors="ignore")
    X_test_u  = X_test_u.drop(columns=cols_to_drop, errors="ignore")

    # ============================
    # 3) Conserver uniquement numériques
    # ============================
    num_cols_u = X_train_u.select_dtypes(include=["number"]).columns

    X_train_u = X_train_u[num_cols_u]
    X_val_u   = X_val_u[num_cols_u]
    X_test_u  = X_test_u[num_cols_u]

    # ============================
    # 4) Normalisation StandardScaler
    # ============================
    scaler_u = StandardScaler()
    scaler_u.fit(X_train_u)

    X_train_u_mlp = scaler_u.transform(X_train_u)
    X_val_u_mlp   = scaler_u.transform(X_val_u)
    X_test_u_mlp  = scaler_u.transform(X_test_u)

    print("✨ Préparation UNSW pour MLP terminée.")
    print(f"Features utilisés : {len(num_cols_u)} colonnes numériques.")

    return (
        X_train_u_mlp,
        X_val_u_mlp,
        X_test_u_mlp,
        y_train_u,
        y_val_u,
        y_test_u,
        num_cols_u,
        scaler_u
    )

def prepare_unsw_balanced_for_mlp(X_train_bal, y_train_bal, num_cols_u, scaler_u):
    """
    Prépare les données équilibrées (oversampling, undersampling, SMOTE) 
    pour UNSW-NB15 avant l'entraînement MLP.

    Paramètres
    ----------
    X_train_bal : DataFrame
        Données d'entraînement équilibrées (X_train oversampled/undersampled/smote)
    y_train_bal : DataFrame ou Series
        Labels équilibrés
    num_cols_u : liste
        Liste des colonnes numériques utilisées dans la baseline UNSW
    scaler_u : StandardScaler
        Scaler déjà fit sur X_train_u original

    Retour
    ------
    X_train_bal_norm : ndarray
        Données équilibrées normalisées
    y_train_bal_bin : Series
        Labels propres (0/1)
    """

    # --- 1) Convertir en Series si DataFrame ---
    if isinstance(y_train_bal, pd.DataFrame):
        y_train_bal = y_train_bal.iloc[:, 0]

    # (UNSW est déjà 0/1 -> pas de binarisation)
    y_train_bal_bin = y_train_bal.astype(int)

    # --- 2) Garder les mêmes colonnes numériques que la baseline ---
    X_train_bal = X_train_bal[num_cols_u]

    # --- 3) Normalisation avec le scaler ORIGINAL ---
    X_train_bal_norm = scaler_u.transform(X_train_bal)

    return X_train_bal_norm, y_train_bal_bin




def load_mlp_model(model_path):
    """
    Charge un modèle MLPBaseline de manière 100% automatique :
      - détecte input_dim depuis les poids
      - choisit 'cuda' si dispo sinon 'cpu'
      - renvoie un modèle prêt pour l'inférence ou FGSM
    """

    # -------------------------------------------------------
    # 1) Device automatique
    # -------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Device utilisé : {device}")

    # -------------------------------------------------------
    # 2) Chargement brut des poids (pas encore dans le modèle)
    # -------------------------------------------------------
    state_dict = torch.load(model_path, map_location=device)

    # -------------------------------------------------------
    # 3) Détection automatique du input_dim
    # -------------------------------------------------------
    # La 1ère couche = 'model.0.weight' dans ton architecture
    first_layer_weight = state_dict["model.0.weight"]
    input_dim = first_layer_weight.shape[1]

    print(f"🔍 input_dim détecté automatiquement : {input_dim}")

    # -------------------------------------------------------
    # 4) Reconstruction du MLPBaseline
    # -------------------------------------------------------
    model = MLPBaseline(input_dim=input_dim).to(device)

    # -------------------------------------------------------
    # 5) Chargement des poids
    # -------------------------------------------------------
    model.load_state_dict(state_dict)
    model.eval()

    print(f"📥 Modèle MLP chargé depuis : {model_path}")

    return model, device
