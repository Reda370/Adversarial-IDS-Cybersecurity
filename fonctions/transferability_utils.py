import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def evaluate_transferability_RF(
        rf_model_path,
        X_clean_path,
        y_clean_path,
        X_adv_path,
        save_dir=None
    ):
    """
    Évalue la transférabilité d'une attaque (ex: FGSM/Substitute) sur le RF.

    - rf_model_path : chemin du modèle RandomForest (.pkl)
    - X_clean_path  : chemin du X_test propre (CSV)
    - y_clean_path  : chemin du y_test (CSV, avec 'BENIGN' / attaques)
    - X_adv_path    : chemin du X_adv (npy)
    - save_dir      : si non None, sauvegarde aussi X_adv dans ce dossier
                      sous le nom 'X_adv_substitute.npy'
    """

    # --- Charger RF ---
    rf = joblib.load(rf_model_path)

    # --- Charger données clean ---
    X_clean = pd.read_csv(X_clean_path).values

    # Binarisation y_clean
    y_clean_raw = pd.read_csv(y_clean_path).iloc[:, 0].astype(str)
    y_clean = y_clean_raw.apply(lambda x: 0 if x == "BENIGN" else 1).values

    # --- Charger données adversariales ---
    X_adv = np.load(X_adv_path)

    # --- Option : sauvegarder X_adv dans save_dir ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        adv_save_path = os.path.join(save_dir, "X_adv_substitute.npy")
        np.save(adv_save_path, X_adv)
        print(f"📁 X_adv substitute sauvegardé dans : {adv_save_path}")

    # -------------------------------
    # 🔵 Évaluation sur données propres
    # -------------------------------
    preds_clean_raw = rf.predict(X_clean)
    preds_clean = np.array([0 if p == "BENIGN" else 1 for p in preds_clean_raw])

    clean_metrics = {
        "accuracy": accuracy_score(y_clean, preds_clean),
        "recall": recall_score(y_clean, preds_clean),
        "precision": precision_score(y_clean, preds_clean),
        "f1": f1_score(y_clean, preds_clean),
        "FNR": 1 - recall_score(y_clean, preds_clean)
    }

    # -------------------------------
    # 🔴 Évaluation sous attaque (X_adv)
    # -------------------------------
    preds_adv_raw = rf.predict(X_adv)
    preds_adv = np.array([0 if p == "BENIGN" else 1 for p in preds_adv_raw])

    adv_metrics = {
        "accuracy": accuracy_score(y_clean, preds_adv),
        "recall": recall_score(y_clean, preds_adv),
        "precision": precision_score(y_clean, preds_adv),
        "f1": f1_score(y_clean, preds_adv),
        "FNR": 1 - recall_score(y_clean, preds_adv)
    }

    return {
        "clean": clean_metrics,
        "adv": adv_metrics
    }


def save_transferability_results(results, save_dir, epsilon=None):
    """
    Sauvegarde les résultats de transférabilité dans un dossier :
    - clean_metrics.json
    - adv_metrics.json
    - metrics.csv  (clean + adv)
    - log.txt      (résumé lisible)
    """

    os.makedirs(save_dir, exist_ok=True)

    # --- Sauvegarde JSON ---
    with open(os.path.join(save_dir, "clean_metrics.json"), "w") as f:
        json.dump(results["clean"], f, indent=4)

    with open(os.path.join(save_dir, "adv_metrics.json"), "w") as f:
        json.dump(results["adv"], f, indent=4)

    # --- Sauvegarde CSV ---
    df = pd.DataFrame({
        "metric": list(results["clean"].keys()),
        "clean": list(results["clean"].values()),
        "adv": list(results["adv"].values())
    })
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    # --- Sauvegarde LOG ---
    with open(os.path.join(save_dir, "log.txt"), "w") as f:
        f.write("=== Résultats Transférabilité RF ===\n")
        if epsilon is not None:
            f.write(f"(epsilon FGSM = {epsilon})\n\n")

        f.write("--- Données propres ---\n")
        for k, v in results["clean"].items():
            f.write(f"{k} : {v}\n")

        f.write("\n--- Sous attaque (X_adv) ---\n")
        for k, v in results["adv"].items():
            f.write(f"{k} : {v}\n")

    print(f"\n📁 Résultats sauvegardés dans : {save_dir}\n")
