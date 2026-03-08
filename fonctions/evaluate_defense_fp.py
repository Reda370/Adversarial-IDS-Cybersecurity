import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 🔹 Loader modèle
# ============================================================

def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return joblib.load(f)


# ============================================================
# ⭐ CICIDS — taux d’évasion + matrices de confusion + TXT ⭐
# ============================================================

def evaluate_fp_pipeline(
    baseline_model_path,
    defended_model_path,
    X_test_path,
    y_test_path,
    X_adv_path,
    dataset="CICIDS",
    save_dir=None
):

    # --- Chargement ---
    X_test = pd.read_csv(X_test_path).values
    y_test = pd.read_csv(y_test_path).iloc[:, 0].values
    X_adv = np.load(X_adv_path)

    # --- Binarisation ---
    if dataset.upper() == "CICIDS":
        y_test_bin = np.where(y_test == "BENIGN", 0, 1)
    else:
        y_test_bin = y_test.astype(int)

    # --- Modèles ---
    baseline = load_model(baseline_model_path)
    defended = load_model(defended_model_path)

    # ============================================================
    # Fonction interne métriques
    # ============================================================
    def compute_metrics(model, X, y_bin):

        y_pred = model.predict(X)

        if dataset.upper() == "CICIDS":
            y_pred_bin = np.where(y_pred == "BENIGN", 0, 1)
        else:
            y_pred_bin = y_pred.astype(int)

        tn, fp, fn, tp = confusion_matrix(y_bin, y_pred_bin).ravel()

        return {
            "accuracy": accuracy_score(y_bin, y_pred_bin),
            "recall": recall_score(y_bin, y_pred_bin),
            "f1": f1_score(y_bin, y_pred_bin),
            "false_negatives": int(fn),
            "preds": y_pred_bin
        }

    # --- Calcul des métriques ---
    baseline_clean = compute_metrics(baseline, X_test, y_test_bin)
    baseline_adv   = compute_metrics(baseline, X_adv,  y_test_bin)
    defended_adv   = compute_metrics(defended, X_adv,  y_test_bin)

    # ============================================================
    # ⭐ Calcul du taux d’évasion (CICIDS uniquement)
    # ============================================================
    if dataset.upper() == "CICIDS":
        total_malicious = np.sum(y_test_bin == 1)
        evasion_base = baseline_adv["false_negatives"] / total_malicious
        evasion_def  = defended_adv["false_negatives"] / total_malicious
    else:
        evasion_base = "N/A"
        evasion_def  = "N/A"

    # ============================================================
    # ⭐ Delta = Defended FP – Baseline FP
    # ============================================================
    def delta(def_val, base_val):
        if isinstance(def_val, (int, float)) and isinstance(base_val, (int, float)):
            d = def_val - base_val
            return f"{'+' if d>=0 else ''}{d:.4f}"
        return "N/A"

    delta_row = {
        "accuracy": delta(defended_adv["accuracy"], baseline_adv["accuracy"]),
        "recall": delta(defended_adv["recall"], baseline_adv["recall"]),
        "f1": delta(defended_adv["f1"], baseline_adv["f1"]),
        "false_negatives": delta(defended_adv["false_negatives"], baseline_adv["false_negatives"]),
        "evasion_rate": delta(evasion_def, evasion_base) if dataset.upper()=="CICIDS" else "N/A",
    }

    # ============================================================
    # ⭐ Tableau final CICIDS
    # ============================================================
    df = pd.DataFrame([
        [
            "Baseline – Clean",
            baseline_clean["accuracy"],
            baseline_clean["recall"],
            baseline_clean["f1"],
            baseline_clean["false_negatives"],
            evasion_base
        ],
        [
            "Baseline – FP",
            baseline_adv["accuracy"],
            baseline_adv["recall"],
            baseline_adv["f1"],
            baseline_adv["false_negatives"],
            evasion_base
        ],
        [
            "Defended – FP",
            defended_adv["accuracy"],
            defended_adv["recall"],
            defended_adv["f1"],
            defended_adv["false_negatives"],
            evasion_def
        ],
        [
            "Delta (Def FP - Base FP)",
            delta_row["accuracy"],
            delta_row["recall"],
            delta_row["f1"],
            delta_row["false_negatives"],
            delta_row["evasion_rate"]
        ],
    ], columns=["Model", "accuracy", "recall", "f1", "false_negatives", "evasion_rate"])

    # ============================================================
    # ⭐ Matrices de confusion (CICIDS)
    # ============================================================
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        def save_cm(name, y_t, y_p):
            cm = confusion_matrix(y_t, y_p)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title(name)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}.png"))
            plt.close()

        if dataset.upper() == "CICIDS":
            save_cm("cm_clean", y_test_bin, baseline_clean["preds"])
            save_cm("cm_adv_baseline", y_test_bin, baseline_adv["preds"])
            save_cm("cm_adv_defended", y_test_bin, defended_adv["preds"])

    # ============================================================
    # ⭐ Export CSV + TXT CICIDS
    # ============================================================
    if save_dir is not None:

        csv_path = os.path.join(save_dir, "comparison_feature_defense.csv")
        df.to_csv(csv_path, index=False)

        txt_path = os.path.join(save_dir, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:

            f.write("===== CICIDS — Évaluation Défense =====\n\n")

            f.write("Baseline Clean:\n")
            for k,v in baseline_clean.items():
                if k!="preds":
                    f.write(f"  {k}: {v}\n")

            f.write("\nBaseline After Attack:\n")
            for k,v in baseline_adv.items():
                if k!="preds":
                    f.write(f"  {k}: {v}\n")
            f.write(f"  Taux d'évasion: {evasion_base}\n")

            f.write("\nDefended After Attack:\n")
            for k,v in defended_adv.items():
                if k!="preds":
                    f.write(f"  {k}: {v}\n")
            f.write(f"  Taux d'évasion: {evasion_def}\n")

            f.write("\nDelta (Def FP - Base FP):\n")
            for k,v in delta_row.items():
                f.write(f"  {k}: {v}\n")

            f.write("\nMatrices de confusion générées:\n")
            f.write("  - cm_clean.png\n")
            f.write("  - cm_adv_baseline.png\n")
            f.write("  - cm_adv_defended.png\n")

        print(f"✔ CICIDS CSV + TXT sauvegardés dans : {save_dir}")

    return df



# ============================================================
# ⭐ UNSW — PAS DE taux d’évasion (mais CM + TXT) ⭐
# ============================================================

def evaluate_fp_pipeline_unsw(
    baseline_model_path,
    defended_model_path,
    X_test_path,
    y_test_path,
    X_adv_path,
    save_dir=None
):

    # --- Chargement ---
    X_test = pd.read_csv(X_test_path).values
    y_test = pd.read_csv(y_test_path).iloc[:, 0].values
    X_adv = np.load(X_adv_path)

    y_test_bin = y_test.astype(int)

    baseline = load_model(baseline_model_path)
    defended = load_model(defended_model_path)

    # ------------------------------------------------------------
    # Fonction interne UNSW
    # ------------------------------------------------------------
    def compute_metrics_unsw(model, X, y_bin):
        y_pred = model.predict(X).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, y_pred).ravel()

        return {
            "accuracy": accuracy_score(y_bin, y_pred),
            "recall": recall_score(y_bin, y_pred),
            "f1": f1_score(y_bin, y_pred),
            "false_positives": fp,
            "false_negatives": fn,
            "preds": y_pred
        }

    baseline_clean = compute_metrics_unsw(baseline, X_test, y_test_bin)
    baseline_adv   = compute_metrics_unsw(baseline, X_adv,  y_test_bin)
    defended_adv   = compute_metrics_unsw(defended, X_adv,  y_test_bin)

    # ------------------------------------------------------------
    # DELTA UNSW
    # ------------------------------------------------------------
    def delta(def_val, base_val):
        d = def_val - base_val
        return f"{'+' if d>=0 else ''}{d:.4f}"

    delta_row = {
        "accuracy": delta(defended_adv["accuracy"], baseline_adv["accuracy"]),
        "recall": delta(defended_adv["recall"], baseline_adv["recall"]),
        "f1": delta(defended_adv["f1"], baseline_adv["f1"]),
        "false_positives": delta(defended_adv["false_positives"], baseline_adv["false_positives"]),
        "false_negatives": delta(defended_adv["false_negatives"], baseline_adv["false_negatives"]),
    }

    # ------------------------------------------------------------
    # Tableau final UNSW
    # ------------------------------------------------------------
    df = pd.DataFrame([
        ["Baseline – Clean",
         baseline_clean["accuracy"], baseline_clean["recall"], baseline_clean["f1"],
         baseline_clean["false_positives"], baseline_clean["false_negatives"]],

        ["Baseline – FP",
         baseline_adv["accuracy"], baseline_adv["recall"], baseline_adv["f1"],
         baseline_adv["false_positives"], baseline_adv["false_negatives"]],

        ["Defended – FP",
         defended_adv["accuracy"], defended_adv["recall"], defended_adv["f1"],
         defended_adv["false_positives"], defended_adv["false_negatives"]],

        ["Delta (Def FP - Base FP)",
         delta_row["accuracy"], delta_row["recall"], delta_row["f1"],
         delta_row["false_positives"], delta_row["false_negatives"]],
    ], columns=["Model", "accuracy", "recall", "f1", "false_positives", "false_negatives"])


    # ------------------------------------------------------------
    # Matrices de confusion (UNSW)
    # ------------------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        def save_cm(name, yt, yp):
            cm = confusion_matrix(yt, yp)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title(name)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}.png"))
            plt.close()

        save_cm("cm_clean", y_test_bin, baseline_clean["preds"])
        save_cm("cm_adv_baseline", y_test_bin, baseline_adv["preds"])
        save_cm("cm_adv_defended", y_test_bin, defended_adv["preds"])


    # ------------------------------------------------------------
    # TXT UNSW
    # ------------------------------------------------------------
    if save_dir:
        txt_path = os.path.join(save_dir, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:

            f.write("===== UNSW — Évaluation Défense =====\n\n")

            f.write("Baseline Clean:\n")
            for k,v in baseline_clean.items():
                if k!="preds":
                    f.write(f"  {k}: {v}\n")

            f.write("\nBaseline Attack:\n")
            for k,v in baseline_adv.items():
                if k!="preds":
                    f.write(f"  {k}: {v}\n")

            f.write("\nDefended Attack:\n")
            for k,v in defended_adv.items():
                if k!="preds":
                    f.write(f"  {k}: {v}\n")

            f.write("\nDelta (Def FP - Base FP):\n")
            for k,v in delta_row.items():
                f.write(f"  {k}: {v}\n")

            f.write("\nMatrices de confusion générées:\n")
            f.write("  - cm_clean.png\n")
            f.write("  - cm_adv_baseline.png\n")
            f.write("  - cm_adv_defended.png\n")

        print(f"✔ UNSW CSV + TXT sauvegardés : {save_dir}")

    return df
