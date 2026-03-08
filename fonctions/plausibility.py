import numpy as np
import pandas as pd
from pathlib import Path

import numpy as np
import json
import os


def evaluate_plausibility(
        X_clean, 
        X_adv, 
        max_neg=0,
        max_nan=0,
        max_above_max=0,
        l2_threshold=1.0,
        linf_threshold=10.0,
        save_path=None
    ):
    """
    Fonction finale de plausibilité :
    - calcule les stats
    - affiche l'interprétation
    - donne un verdict (réaliste / non-réaliste)
    - SAUVEGARDE les stats dans un JSON si save_path est fourni
    """

    # ------------------------------------------------------------
    # 0) Convertir DataFrame -> numpy
    # ------------------------------------------------------------
    X_clean_np = X_clean.values if isinstance(X_clean, pd.DataFrame) else X_clean
    X_adv_np   = X_adv.values   if isinstance(X_adv, pd.DataFrame)   else X_adv

    # ------------------------------------------------------------
    # 1) Calcul des métriques
    # ------------------------------------------------------------
    neg_mask = (X_adv_np < 0) & (X_clean_np >= 0)
    nb_negative = int(neg_mask.sum())

    max_clean = X_clean_np.max(axis=0)
    nb_above_max = int((X_adv_np > max_clean).sum())

    nb_nan = int(np.isnan(X_adv_np).sum())

    diff = X_adv_np - X_clean_np
    l2_distance_mean = float(np.mean(np.linalg.norm(diff, axis=1)))
    l_inf_max = float(np.max(np.abs(diff)))

    # dictionnaire de stats
    results = {
        "l2_distance_mean": l2_distance_mean,
        "l_inf_max": l_inf_max,
        "nb_negative": nb_negative,
        "nb_nan": nb_nan,
        "nb_above_max": nb_above_max
    }

    # ------------------------------------------------------------
    # 2) Décision réaliste / non réaliste
    # ------------------------------------------------------------
    is_realistic = True

    if nb_negative > max_neg:
        is_realistic = False
    if nb_nan > max_nan:
        is_realistic = False
    if nb_above_max > max_above_max:
        is_realistic = False
    if l2_distance_mean > l2_threshold:
        is_realistic = False
    if l_inf_max > linf_threshold:
        is_realistic = False

    # ------------------------------------------------------------
    # 3) Affichage
    # ------------------------------------------------------------
    print("\n🔍 Résultats du contrôle de plausibilité :")
    print(results)

    print("\n🧠 Interprétation :")

    if nb_negative > 0:
        print(f"⚠️ {nb_negative} valeurs négatives : non plausible.")
    if nb_nan > 0:
        print(f"⚠️ {nb_nan} NaN détectés.")
    if nb_above_max > 0:
        print(f"⚠️ {nb_above_max} valeurs dépassent le max du dataset.")

    print(f"• L2 distance moyenne : {l2_distance_mean:.4f}")
    print(f"• L_inf max           : {l_inf_max:.4f}")

    if is_realistic:
        print("\n✔ Attaque réaliste.")
    else:
        print("\n❌ Attaque NON réaliste.")

    # ------------------------------------------------------------
    # 4) Sauvegarde des résultats dans un JSON
    # ------------------------------------------------------------
    if save_path is not None:
        folder = os.path.dirname(save_path)
        os.makedirs(folder, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(
                {"results": results, "is_realistic": is_realistic},
                f,
                indent=4
            )
        print(f"\n📁 Stats sauvegardées dans : {save_path}")

    return results, is_realistic



def compute_plausibility_stats(X_adv, X_original, save_path=None):
    """
    Calcule les statistiques de plausibilité des exemples adversariaux.

    Paramètres
    ----------
    X_adv : ndarray ou DataFrame
        Données adversariales générées.
    X_original : ndarray ou DataFrame
        Données originales X_test utilisées pour l'attaque.
    save_path : str ou None
        Si fourni, sauvegarde les stats dans un fichier JSON.

    Retourne
    --------
    stats : dict
        Dictionnaire contenant toutes les métriques utiles.
    """

    X_adv_np = np.array(X_adv)
    X_orig_np = np.array(X_original)

    delta = X_adv_np - X_orig_np

    # Statistiques
    stats = {
        "l2_mean": float(np.mean(np.linalg.norm(delta, axis=1))),
        "l_inf_max": float(np.max(np.abs(delta))),
        "nb_negatives": int(np.sum(X_adv_np < 0)),
        "nb_nan": int(np.isnan(X_adv_np).sum()),
        "nb_above_max": int(np.sum(X_adv_np > 1e12)),  # sécurité dataset
    }

    # Sauvegarde optionnelle
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"📁 Stats de plausibilité sauvegardées dans : {save_path}")

    return stats




def check_plausibility_RF(
    X_clean,
    X_adv,
    min_vals,
    max_vals,
    output_dir=None,
    l2_threshold=None,
    linf_threshold=None
):
    """
    Contrôle de plausibilité universel CICIDS / UNSW.
    Seuils dynamiques selon dataset + type de balancing.
    """

    X_clean = np.asarray(X_clean, dtype=float)
    X_adv   = np.asarray(X_adv, dtype=float)

    # =============== Détection dataset ==================
    n_features = X_clean.shape[1]

    if n_features == 78:
        dataset = "CICIDS"
    elif n_features == 187:
        dataset = "UNSW"
    else:
        dataset = "UNKNOWN"

    print(f"📌 Dataset détecté pour plausibility : {dataset}")

    # =============== Seuils automatiques ==================
    path_str = str(output_dir).lower()

    if dataset == "CICIDS":

        # NO BALANCING
        if "no_balancing" in path_str:
            l2_threshold  = 10   if l2_threshold is None else l2_threshold
            linf_threshold = 120 if linf_threshold is None else linf_threshold

        # SMOTE
        elif "smote" in path_str:
            l2_threshold  = 20   if l2_threshold is None else l2_threshold
            linf_threshold = 150 if linf_threshold is None else linf_threshold

        # UNDERSAMPLING / OVERSAMPLING
        elif "under" in path_str or "over" in path_str:
            l2_threshold  = 15   if l2_threshold is None else l2_threshold
            linf_threshold = 120 if linf_threshold is None else linf_threshold

        # fallback CICIDS
        else:
            l2_threshold  = 10
            linf_threshold = 100

    elif dataset == "UNSW":
        l2_threshold  = 20   if l2_threshold is None else l2_threshold
        linf_threshold = 600 if linf_threshold is None else linf_threshold

    else:
        l2_threshold  = 10
        linf_threshold = 200

    print(f"✔️ Seuil L2 utilisé    : {l2_threshold}")
    print(f"✔️ Seuil L∞ utilisé    : {linf_threshold}")

    # =============== Distances ==================
    diff = X_adv - X_clean
    l2_mean = float(np.mean(np.linalg.norm(diff, axis=1)))
    linf_max = float(np.max(np.abs(diff)))

    # =============== Vérifications ==================
    nb_negative = int((X_adv < 0).sum())
    nb_nan = int(np.isnan(X_adv).sum())

    nb_above_max = int((X_adv > max_vals).sum())

    stats = {
        "l2_distance_mean": l2_mean,
        "l_inf_max": linf_max,
        "nb_negative": nb_negative,
        "nb_nan": nb_nan,
        "nb_above_max": nb_above_max,
    }

    print("\n🔍 Résultats du contrôle de plausibilité :")
    print(stats)

    # =============== Verdict ==================
    is_realistic = True

    if nb_negative > 0: is_realistic = False
    if nb_above_max > 0: is_realistic = False
    if nb_nan > 0: is_realistic = False
    if l2_mean > l2_threshold: is_realistic = False
    if linf_max > linf_threshold: is_realistic = False

    print("\n📢 VERDICT DE PLAUSIBILITÉ")
    print("---------------------------")
    if is_realistic:
        print("✅ L’attaque est RÉALISTE.")
        verdict = "REALISTIC"
    else:
        print("❌ L’attaque n’est PAS réaliste.")
        verdict = "NOT_REALISTIC"

    # ============== Sauvegarde ==================
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "plausibility.json", "w") as f:
            json.dump({"stats": stats, "verdict": verdict}, f, indent=4)

        with open(out / "plausibility.txt", "w") as f:
            f.write("=== PLAUSIBILITY REPORT ===\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nVERDICT: {verdict}\n")

        print(f"\n📁 Rapport sauvegardé dans : {out}")

    return stats, is_realistic
