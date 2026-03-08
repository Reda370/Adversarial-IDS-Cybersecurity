import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score

def get_immutable_mask_unsw(columns):
    """
    Masque immuable pour UNSW-NB15 :
    0 = non modifiable
    1 = modifiable
    """

    immutable_mask = np.ones(len(columns), dtype=int)

    for i, col in enumerate(columns):
        col_low = col.lower()

        # Protocol one-hot
        if col_low.startswith("proto_"):
            immutable_mask[i] = 0

        # Services one-hot
        if col_low.startswith("service_"):
            immutable_mask[i] = 0

        # States one-hot
        if col_low.startswith("state_"):
            immutable_mask[i] = 0

        # Immuables liés au protocole TCP
        if col_low in ["swin", "dwin", "stcpb", "dtcpb"]:
            immutable_mask[i] = 0

        # Profondeur protocol / analyse HTTP/FTP
        if col_low in ["trans_depth", "is_ftp_login", "ct_ftp_cmd", "is_sm_ips_ports"]:
            immutable_mask[i] = 0

    return immutable_mask


def get_immutable_mask_for_cicids_final(columns):
    """
    Génère un masque immuable (0 = non modifiable, 1 = modifiable)
    basé sur les colonnes CICIDS.
    """

    # mots-clés des features non modifiables
    immutable_keywords = [
        "flag",
        "header",
        "subflow",
        "fwd packets",
        "bwd packets",
        "fwd bytes",
        "bwd bytes",
        "flow duration",
        "win",
        "destination port",
        "port",
        "min seg",

        # === 🔥 AJOUT INDISPENSABLE POUR CORRIGER L∞ !!! ===
        "iat",               # Flow IAT Max, Total, Mean, Bwd/Fwd IAT
        "packets/s",         # Fwd Packets/s, Bwd Packets/s
        "bytes/s",           # Flow Bytes/s
        "fwd pkt",           # Tot Fwd Pkts
        "bwd pkt"            # Tot Bwd Pkts
    ]

    mask = np.ones(len(columns), dtype=int)

    for i, col in enumerate(columns):
        col_low = col.lower().strip()

        # match par mots-clés
        for k in immutable_keywords:
            if k in col_low:
                mask[i] = 0
                break

    return mask


def run_fgsm_finale(
    model,
    X_test_np,
    y_test_np,
    scaler,
    min_vals,
    max_vals,
    immutable_mask,
    epsilon=0.01,
    max_ratio=0.05,
    batch_size=1024,
    device=None,
    output_dir="../attacks/FGSM_realistic"
):
    """
    FGSM réaliste FINAL :
    - FGSM dans l'espace normalisé
    - Dénormalisation → clamp min/max TRAIN
    - Variation max ±max_ratio (ex: 5%)
    - Masque FEATURES immuables
    - Renormalisation finale
    """

    # ============================================================
    # 0) DEVICE
    # ============================================================
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # ============================================================
    # 1) NORMALISATION
    # ============================================================
    X_clean_norm = scaler.transform(X_test_np)
    X_clean = torch.tensor(X_clean_norm, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test_np, dtype=torch.long, device=device)

    X_clean.requires_grad_(True)
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # 2) FGSM DANS L'ESPACE NORMALISÉ
    # ============================================================
    logits = model(X_clean)
    loss = criterion(logits, y_test)

    model.zero_grad()
    loss.backward()

    grad_sign = X_clean.grad.data.sign()

    X_adv_norm = X_clean + epsilon * grad_sign
    X_adv_norm = X_adv_norm.detach().cpu().numpy()

    # ============================================================
    # 3) DÉNORMALISATION
    # ============================================================
    X_adv_real = scaler.inverse_transform(X_adv_norm)
    X_clean_real = scaler.inverse_transform(X_clean_norm)

    # ============================================================
    # 4) CLAMP AVEC MIN/MAX DU TRAIN (IMPORTANT !!!)
    # ============================================================
    X_adv_real = np.clip(X_adv_real, min_vals, max_vals)

    # ============================================================
    # 5) LIMITATION VARIATION RELATIVE (REALISME)
    # ============================================================
    delta = X_adv_real - X_clean_real
    limit = max_ratio * (np.abs(X_clean_real) + 1e-9)  # éviter 0
    delta = np.clip(delta, -limit, limit)
    X_adv_real = X_clean_real + delta

    # ============================================================
    # 6) MASQUE DES FEATURES IMMUTABLES
    # ============================================================
    X_adv_real[:, immutable_mask == 0] = X_clean_real[:, immutable_mask == 0]

    # ============================================================
    # 7) RENORMALISATION POUR LE MLP
    # ============================================================
    X_adv_norm = scaler.transform(X_adv_real)
    X_adv_tensor = torch.tensor(X_adv_norm, dtype=torch.float32, device=device)

    # ============================================================
    # 8) PRÉDICTIONS ADVERSARIALES
    # ============================================================
    with torch.no_grad():
        preds_adv = model(X_adv_tensor).argmax(1).cpu().numpy()

    # ============================================================
    # 9) SAUVEGARDE
    # ============================================================
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/X_adv_original.npy", X_adv_real)
    np.save(f"{output_dir}/X_adv_normalized.npy", X_adv_norm)
    print(f"📁 Fichiers sauvegardés dans : {output_dir}")

    # ============================================================
    # 9) ÉVALUATION CLEAN & ADVERSARIALE
    # ============================================================
    # Prédictions clean
    X_clean_tensor = torch.tensor(X_clean_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred_clean = model(X_clean_tensor).argmax(1).cpu().numpy()
    
    # Prédictions adversariales (déjà dans preds_adv)
    y_pred_adv = preds_adv
    
    # Métriques
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    
    acc_clean  = accuracy_score(y_test_np, y_pred_clean)
    acc_adv    = accuracy_score(y_test_np, y_pred_adv)
    rec_clean  = recall_score(y_test_np, y_pred_clean)
    rec_adv    = recall_score(y_test_np, y_pred_adv)
    f1_clean   = f1_score(y_test_np, y_pred_clean)
    f1_adv     = f1_score(y_test_np, y_pred_adv)
    
    fn_clean   = int(((y_test_np == 1) & (y_pred_clean == 0)).sum())
    fn_adv     = int(((y_test_np == 1) & (y_pred_adv == 0)).sum())
    fn_diff    = fn_adv - fn_clean
    
    print("\n📊 ÉVALUATION FGSM")
    print("--------------------------------------------")
    print(f"Accuracy clean  : {acc_clean:.4f}")
    print(f"Accuracy adv    : {acc_adv:.4f}")
    print(f"Recall clean    : {rec_clean:.4f}")
    print(f"Recall adv      : {rec_adv:.4f}")
    print(f"F1-score clean  : {f1_clean:.4f}")
    print(f"F1-score adv    : {f1_adv:.4f}")
    
    print("\n🔥 Faux négatifs")
    print(f"FN clean : {fn_clean}")
    print(f"FN adv   : {fn_adv}")
    print(f"➡️  Augmentation des FN : {fn_diff}")



    return X_adv_real, X_adv_norm, preds_adv
