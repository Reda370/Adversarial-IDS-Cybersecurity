# ============================================================
# adversarial_training.py – US20 Défense FGSM pour MLPBaseline
# Auto CICIDS (BENIGN vs attaque) / UNSW (0/1)
# Compatible baseline binaire existante (num_classes=2)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from fonctions.MLP_baseline import MLPBaseline


# ------------------------------------------------------------
# 1) FGSM pour l'entraînement (avec bruit optionnel)
# ------------------------------------------------------------
def fgsm_for_training(model, x, y, epsilon, noise_ratio=0.0):
    """
    FGSM simple et efficace pour l'adversarial training.
    - epsilon : intensité FGSM
    - noise_ratio : bruit pour éviter loss=0
    """
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)

    model.zero_grad()
    loss.backward()

    grad = x_adv.grad.sign()
    noise = noise_ratio * torch.randn_like(x_adv)

    x_adv = x_adv + epsilon * grad + noise

    return x_adv.detach()


# ------------------------------------------------------------
# 2) Charger le MLP baseline existant (binaire)
# ------------------------------------------------------------
def load_mlp_baseline(model_path, input_dim):
    """
    Charge ton modèle baseline binaire EXACT (num_classes=2)
    """
    model = MLPBaseline(input_dim=input_dim, num_classes=2)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------
# 3) Entraînement adversarial + sauvegarde modèle défendu
# ------------------------------------------------------------
def adversarial_training_and_save(
        model_path,
        X_train, y_train,
        X_val, y_val,
        save_path,
        epsilon=0.15,      # tu peux adapter par dataset dans le main
        adv_ratio=0.5,
        noise_ratio=0.01,
        nb_epochs=10,
        batch_size=256,
        lr=1e-3
    ):

    # ============================================================
    # 0) Conversion DataFrame → numpy si nécessaire
    # ============================================================
    if hasattr(X_train, "values"): X_train = X_train.values
    if hasattr(X_val,   "values"): X_val   = X_val.values
    if hasattr(y_train, "values"): y_train = y_train.values
    if hasattr(y_val,   "values"): y_val   = y_val.values

    # Flatten labels
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if y_val.ndim > 1:
        y_val = y_val.ravel()

    # Copies pour analyse
    y_train_raw = y_train.copy()
    y_val_raw   = y_val.copy()

    # ============================================================
    # 1) Détection dataset & binarisation
    # ============================================================
    dataset = "UNKNOWN"

    # Vue string
    y_train_str = y_train_raw.astype(str)
    y_val_str   = y_val_raw.astype(str)

    uniques_str = np.unique(y_train_str)

    if "BENIGN" in uniques_str:
        # ==== CICIDS ====
        dataset = "CICIDS"
        y_train_bin = (y_train_str != "BENIGN").astype(int)
        y_val_bin   = (y_val_str   != "BENIGN").astype(int)

    else:
        # On tente numérique
        try:
            y_train_num = y_train_raw.astype(int)
            y_val_num   = y_val_raw.astype(int)
            uniques_num = np.unique(y_train_num)

            if set(uniques_num).issubset({0, 1}):
                # ==== UNSW (0/1) ====
                dataset = "UNSW"
                y_train_bin = y_train_num
                y_val_bin   = y_val_num
            else:
                # Autre dataset numérique : classe majoritaire = benign
                dataset = "NUMERIC_OTHER"
                counts = np.bincount(y_train_num)
                benign_class = np.argmax(counts)
                y_train_bin = (y_train_num != benign_class).astype(int)
                y_val_bin   = (y_val_num   != benign_class).astype(int)
        except Exception:
            # Cas exotique : on choisit la classe string majoritaire
            dataset = "STRING_OTHER"
            uniq, counts = np.unique(y_train_str, return_counts=True)
            benign_class = uniq[np.argmax(counts)]
            y_train_bin = (y_train_str != benign_class).astype(int)
            y_val_bin   = (y_val_str   != benign_class).astype(int)

    print(f"📌 Dataset détecté pour binarisation : {dataset}")
    # (optionnel) vérif rapide
    # print("Labels binaires :", np.unique(y_train_bin, return_counts=True))

    # ============================================================
    # 2) Device
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 3) input_dim et chargement modèle baseline binaire
    # ============================================================
    input_dim = X_train.shape[1]

    print(f"📥 Chargement du modèle baseline : {model_path}")
    model = load_mlp_baseline(model_path, input_dim).to(device)

    # ============================================================
    # 4) DataLoader
    # ============================================================
    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_bin, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_bin, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print("\n========== US20 – Adversarial Training START ==========\n")

    # ============================================================
    # 5) Boucle d'entraînement adversarial
    # ============================================================
    for epoch in range(1, nb_epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            n = xb.size(0)
            n_adv = int(n * adv_ratio)

            if n_adv > 0:
                xb_clean = xb[:n_adv]
                yb_clean = yb[:n_adv]

                xb_adv = fgsm_for_training(
                    model,
                    xb_clean,
                    yb_clean,
                    epsilon=epsilon,
                    noise_ratio=noise_ratio
                )

                xb_mix = torch.cat([xb, xb_adv], dim=0)
                yb_mix = torch.cat([yb, yb_clean], dim=0)
            else:
                xb_mix, yb_mix = xb, yb

            optimizer.zero_grad()
            preds = model(xb_mix)
            loss = loss_fn(preds, yb_mix)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(1)
            acc = (val_preds == y_val_t).float().mean().item()

        print(f"Epoch {epoch}/{nb_epochs} — Loss={mean_loss:.4f} | ValAcc={acc:.4f}")

    # ============================================================
    # 6) Sauvegarde modèle défendu
    # ============================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\n💾 Modèle défendu sauvegardé dans : {save_path}")

    return model
