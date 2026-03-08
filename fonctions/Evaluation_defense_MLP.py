# ============================================================
# evaluation_defense_MLP.py – Version finale corrigée
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from fonctions.MLP_baseline import MLPBaseline


# ============================================================
# 🔹 Load MLP model
# ============================================================

def load_mlp(model_path, input_dim):
    model = MLPBaseline(input_dim=input_dim, num_classes=2)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# 🔹 Binarize labels (supports CICIDS + UNSW)
# ============================================================

def binarize_labels(y):
    y = np.array(y)
    y_str = y.astype(str)

    # CICIDS
    if "BENIGN" in np.unique(y_str):
        return (y_str != "BENIGN").astype(int)

    # UNSW (already numeric)
    try:
        y_int = y.astype(int)
        if set(np.unique(y_int)).issubset({0, 1}):
            return y_int
    except:
        pass

    # fallback
    return (y_str != y_str[0]).astype(int)


# ============================================================
# 🔹 Evaluate model performance
# ============================================================

def evaluate(model, X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = model(X_t).detach().numpy().argmax(1)

    FN = int(((y == 1) & (preds == 0)).sum())

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "false_negatives": FN,
        "preds": preds
    }


def sign_format(value):
    """Formats numeric delta with sign."""
    if isinstance(value, int):
        return f"{value:+d}"
    return f"{value:+.4f}"


# ============================================================
# 🔥 FINAL FUNCTION — WITH CM + TXT + EVASION RATE + FIX
# ============================================================

def evaluate_defense_delta(
        baseline_model_path,
        defended_model_path,
        X_clean_path,
        y_clean_path,
        X_adv_path,
        save_dir,
        dataset="CICIDS"
    ):

    os.makedirs(save_dir, exist_ok=True)

    # ===== Load Data =====
    X_clean = pd.read_csv(X_clean_path).values
    y_clean = pd.read_csv(y_clean_path).values.ravel()
    X_adv   = np.load(X_adv_path)

    y_bin = binarize_labels(y_clean)
    input_dim = X_clean.shape[1]

    # ===== Load models =====
    baseline = load_mlp(baseline_model_path, input_dim)
    defended = load_mlp(defended_model_path, input_dim)

    # ===== Evaluate =====
    res_clean     = evaluate(baseline, X_clean, y_bin)
    res_base_adv  = evaluate(baseline, X_adv,  y_bin)
    res_def_adv   = evaluate(defended, X_adv,  y_bin)

    # ============================================================
    # ⭐ EVASION RATE (CICIDS only)
    # ============================================================

    if dataset.upper() == "CICIDS":
        total_malicious = np.sum(y_bin == 1)
        evasion_base = res_base_adv["false_negatives"] / total_malicious
        evasion_def  = res_def_adv["false_negatives"] / total_malicious
    else:
        evasion_base = "N/A"
        evasion_def  = "N/A"

    # ============================================================
    # ⭐ CONFUSION MATRICES
    # ============================================================

    def save_cm(name, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}.png"))
        plt.close()

    save_cm("cm_clean", y_bin, res_clean["preds"])
    save_cm("cm_adv_baseline", y_bin, res_base_adv["preds"])
    save_cm("cm_adv_defended", y_bin, res_def_adv["preds"])


    # ============================================================
    # ⭐ TABLEAU DE COMPARAISON (CSV + Delta fixée)
    # ============================================================

    table = pd.DataFrame({
        "Baseline – Clean": {
            "accuracy": res_clean["accuracy"],
            "precision": res_clean["precision"],
            "recall": res_clean["recall"],
            "f1": res_clean["f1"],
            "false_negatives": res_clean["false_negatives"],
            "evasion_rate": evasion_base
        },
        "Baseline – FGSM": {
            "accuracy": res_base_adv["accuracy"],
            "precision": res_base_adv["precision"],
            "recall": res_base_adv["recall"],
            "f1": res_base_adv["f1"],
            "false_negatives": res_base_adv["false_negatives"],
            "evasion_rate": evasion_base
        },
        "Defended – FGSM": {
            "accuracy": res_def_adv["accuracy"],
            "precision": res_def_adv["precision"],
            "recall": res_def_adv["recall"],
            "f1": res_def_adv["f1"],
            "false_negatives": res_def_adv["false_negatives"],
            "evasion_rate": evasion_def
        }
    }).T

    # ============================================================
    # ⭐ DELTA (bug FIX: evasion_rate handled separately)
    # ============================================================

    delta = {}

    # standard metrics delta
    for metric in ["accuracy", "precision", "recall", "f1", "false_negatives"]:
        raw = res_def_adv[metric] - res_base_adv[metric]
        delta[metric] = sign_format(raw)

    # evasion rate delta
    if dataset.upper() == "CICIDS":
        delta["evasion_rate"] = sign_format(evasion_def - evasion_base)
    else:
        delta["evasion_rate"] = "N/A"

    table.loc["Delta (Defended FGSM - Baseline FGSM)"] = delta


    # ============================================================
    # ⭐ SAVE CSV
    # ============================================================

    csv_path = os.path.join(save_dir, "defense_comparative_table_delta.csv")
    table.to_csv(csv_path)
    print(f"💾 Tableau CSV sauvegardé : {csv_path}")


    # ============================================================
    # ⭐ SAVE TXT (rapport)
    # ============================================================

    txt_path = os.path.join(save_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("===== MLP Defense Evaluation =====\n\n")

        f.write("---- Baseline Clean ----\n")
        for k,v in res_clean.items():
            if k!="preds":
                f.write(f"{k}: {v}\n")

        f.write("\n---- Baseline After FGSM ----\n")
        for k,v in res_base_adv.items():
            if k!="preds":
                f.write(f"{k}: {v}\n")
        f.write(f"evasion_rate: {evasion_base}\n")

        f.write("\n---- Defended After FGSM ----\n")
        for k,v in res_def_adv.items():
            if k!="preds":
                f.write(f"{k}: {v}\n")
        f.write(f"evasion_rate: {evasion_def}\n")

        f.write("\n---- Delta ----\n")
        for k,v in delta.items():
            f.write(f"{k}: {v}\n")

        f.write("\nMatrices de confusion générées:\n")
        f.write("  - cm_clean.png\n")
        f.write("  - cm_adv_baseline.png\n")
        f.write("  - cm_adv_defended.png\n")

    print(f"📝 Rapport TXT sauvegardé : {txt_path}")

    return table
