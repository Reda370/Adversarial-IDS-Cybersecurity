import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 🔹 Charge labels & convertit en binaire
# ============================================================

def load_labels_generic(path):
    y = pd.read_csv(path).iloc[:, 0].astype(str).str.strip().str.upper()

    return np.array([
        0 if val in ("BENIGN", "NORMAL", "0", "0.0") else 1
        for val in y
    ], dtype=int)


# ============================================================
# 🔹 Prédiction universelle (RF sklearn + MLP PyTorch)
# ============================================================

def predict_model(model, X):
    if hasattr(model, "predict"):          # sklearn
        return model.predict(X)

    elif hasattr(model, "forward"):        # PyTorch
        import torch
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    else:
        raise ValueError(f"Unknown model type: {type(model)}")


# ============================================================
# 🔹 Convertir prédictions RF textuelles → binaire
# ============================================================

def convert_predictions_to_binary(preds):
    preds = np.array(preds)
    if preds.dtype == object:  # strings (BENIGN, ATTACK, etc.)
        preds = np.array([0 if str(p).upper() == "BENIGN" else 1 for p in preds])
    else:
        preds = preds.astype(int)
    return preds



# ============================================================
# 🔹 FONCTION US16 – baseline vs adversarial + taux d’évasion + matrices confusion
# ============================================================

def evaluate_us16_and_save(
    model,
    X_clean,
    y_clean,
    X_adv,
    y_true,
    save_dir="../results/US16/"
):
    """
    Compare baseline vs adversarial :
      ✓ Accuracy, precision, recall, F1
      ✓ Faux négatifs
      ✓ Taux d’évasion
      ✓ Matrice de confusion (clean + adv)
      ✓ Tableau comparatif Markdown
    """

    # Convert labels
    y_clean = np.asarray(y_clean).astype(int)
    y_true  = np.asarray(y_true).astype(int)

    os.makedirs(save_dir, exist_ok=True)

    # --- Prédictions ---
    preds_clean = convert_predictions_to_binary(predict_model(model, X_clean))
    preds_adv   = convert_predictions_to_binary(predict_model(model, X_adv))

    # --- FN clean et adv ---
    fn_clean = np.sum((preds_clean == 0) & (y_clean == 1))
    fn_adv   = np.sum((preds_adv   == 0) & (y_true  == 1))

    # =====================================================
    # ⭐ Taux d’évasion
    # =====================================================
    total_malicious = np.sum(y_true == 1)
    evasion_rate = fn_adv / total_malicious if total_malicious > 0 else 0.0

    # =====================================================
    # ⭐ Matrices de confusion
    # =====================================================
    cm_clean = confusion_matrix(y_clean, preds_clean)
    cm_adv   = confusion_matrix(y_true, preds_adv)

    # --- Save CM clean ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix – CLEAN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm_clean.png"))
    plt.close()

    # --- Save CM adv ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds')
    plt.title("Confusion Matrix – ADVERSARIAL")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm_adv.png"))
    plt.close()

    # =====================================================
    # ⭐ Calcul des métriques complètes
    # =====================================================
    metrics = {
        "Accuracy": (
            accuracy_score(y_clean, preds_clean),
            accuracy_score(y_true, preds_adv)
        ),
        "Recall": (
            recall_score(y_clean, preds_clean, zero_division=0),
            recall_score(y_true, preds_adv, zero_division=0)
        ),
        "Precision": (
            precision_score(y_clean, preds_clean, zero_division=0),
            precision_score(y_true, preds_adv, zero_division=0)
        ),
        "F1-score": (
            f1_score(y_clean, preds_clean, zero_division=0),
            f1_score(y_true, preds_adv, zero_division=0)
        ),
        "Faux négatifs": (fn_clean, fn_adv),
        "Taux d’évasion": (0.0, evasion_rate)   # 0 pour baseline
    }

    # =====================================================
    # ⭐ Sauvegarde Markdown
    # =====================================================
    md_path = os.path.join(save_dir, "tableau_comparatif.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Tableau comparatif – Baseline vs Adversarial (US16)\n\n")
        f.write("| Métrique | Baseline | Adversarial | Δ |\n")
        f.write("|----------|-----------|-------------|---------|\n")

        for metric, (clean, adv) in metrics.items():
            delta = adv - clean
            f.write(f"| {metric} | {clean:.4f} | {adv:.4f} | {delta:+.4f} |\n")

        f.write("\n---\n")
        f.write(f"### 🔥 Taux d'évasion : **{evasion_rate:.4f}**\n")
        f.write("- Plus il est haut, plus l’attaque est efficace.\n")

        f.write("\n### 🖼️ Matrices de confusion sauvegardées :\n")
        f.write("- cm_clean.png\n")
        f.write("- cm_adv.png\n")

    print(f"📁 Tableau US16 sauvegardé dans : {md_path}")
    print(f"🔥 Taux d’évasion = {evasion_rate:.4f}")
    print("🖼️ cm_clean.png + cm_adv.png générés")

    return metrics, md_path



# ============================================================
# 🔹 Version spéciale UNSW – FP & FN + taux d’évasion + CM
# ============================================================

def evaluate_us16_unsw_and_save_fp_fn(
    model,
    X_clean,
    y_clean,
    X_adv,
    y_true,
    save_dir="../results/US16_UNSW/"
):
    """
    Version UNSW :
      ✓ FP + FN
      ✓ Accuracy, Precision, Recall, F1
      ✓ Taux d’évasion
      ✓ Matrice confusion clean + adv
      ✓ Tableau Markdown
    """

    y_clean = np.asarray(y_clean).astype(int)
    y_true  = np.asarray(y_true).astype(int)

    os.makedirs(save_dir, exist_ok=True)

    preds_clean = convert_predictions_to_binary(predict_model(model, X_clean))
    preds_adv   = convert_predictions_to_binary(predict_model(model, X_adv))

    # --- FP & FN ---
    fp_clean = np.sum((preds_clean == 1) & (y_clean == 0))
    fp_adv   = np.sum((preds_adv   == 1) & (y_true  == 0))

    fn_clean = np.sum((preds_clean == 0) & (y_clean == 1))
    fn_adv   = np.sum((preds_adv   == 0) & (y_true  == 1))

    # --- Taux d’évasion ---
    total_malicious = np.sum(y_true == 1)
    evasion_rate = fn_adv / total_malicious if total_malicious > 0 else 0.0

    # --- Matrices de confusion ---
    cm_clean = confusion_matrix(y_clean, preds_clean)
    cm_adv   = confusion_matrix(y_true, preds_adv)

    # save clean CM
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix – CLEAN")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm_clean.png"))
    plt.close()

    # save adv CM
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds')
    plt.title("Confusion Matrix – ADVERSARIAL")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm_adv.png"))
    plt.close()

    # --- metrics ---
    metrics = {
        "Accuracy": (
            accuracy_score(y_clean, preds_clean),
            accuracy_score(y_true, preds_adv)
        ),
        "Recall": (
            recall_score(y_clean, preds_clean, zero_division=0),
            recall_score(y_true, preds_adv, zero_division=0)
        ),
        "Precision": (
            precision_score(y_clean, preds_clean, zero_division=0),
            precision_score(y_true, preds_adv, zero_division=0)
        ),
        "F1-score": (
            f1_score(y_clean, preds_clean, zero_division=0),
            f1_score(y_true, preds_adv, zero_division=0)
        ),
        "Faux positifs": (fp_clean, fp_adv),
        "Faux négatifs": (fn_clean, fn_adv),
        "Taux d’évasion": (0.0, evasion_rate)
    }

    # --- Save MD ---
    md_path = os.path.join(save_dir, "tableau_comparatif_unsw.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Tableau comparatif UNSW – Baseline vs Adversarial\n\n")
        f.write("| Métrique | Baseline | Adversarial | Δ |\n")
        f.write("|----------|-----------|-------------|---------|\n")

        for metric, (clean, adv) in metrics.items():
            delta = adv - clean
            f.write(f"| {metric} | {clean:.4f} | {adv:.4f} | {delta:+.4f} |\n")

        f.write(f"\n### 🔥 Taux d'évasion : **{evasion_rate:.4f}**\n")
        f.write("\n### 🖼️ Matrices de confusion sauvegardées :\n")
        f.write("- cm_clean.png\n")
        f.write("- cm_adv.png\n")

    print(f"📁 Tableau UNSW sauvegardé dans : {md_path}")
    print(f"🔥 Taux d’évasion = {evasion_rate:.4f}")
    print("🖼️ cm_clean.png + cm_adv.png générés")

    return metrics, md_path
