import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score


def binarize_labels(arr):
    arr = np.asarray(arr)

    if arr.dtype == object or arr.dtype.type is np.str_:
        return np.where(arr == "BENIGN", 0, 1)

    return np.where(arr == 0, 0, 1)



def get_immutable_mask_unsw_RF(cols):
    immutable = np.ones(len(cols), dtype=int)

    for i, c in enumerate(cols):
        name = c.lower()

        # Protocole, services, états (one-hot)
        if name.startswith("proto_"):
            immutable[i] = 0
        if name.startswith("service_"):
            immutable[i] = 0
        if name.startswith("state_"):
            immutable[i] = 0

        # Colonnes booléennes spécifiques
        if c.strip() in ["is_sm_ips_ports", "is_ftp_login"]:
            immutable[i] = 0

        # Colonnes ct_* (comptage discret)
        if c.startswith("ct_"):
            immutable[i] = 0

    return immutable

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score


# =============================
# Dataset auto-detection
# =============================
def detect_dataset_from_columns(columns):
    cols = [c.lower() for c in columns]

    # CICIDS specific signatures
    cicids_sigs = ["flow bytes/s", "flow packets/s", "flow duration"]
    if any(sig in c for c in cols for sig in cicids_sigs):
        return "CICIDS"

    # UNSW specific signatures
    if any(c.startswith("ct_") for c in cols):
        return "UNSW"

    return "UNKNOWN"


# =============================
# Label binarizer CICIDS/UNSW
# =============================
def binarize_labels(arr):
    arr = np.asarray(arr)

    if arr.dtype == object or arr.dtype.type is np.str_:
        return np.where(arr == "BENIGN", 0, 1)

    return np.where(arr == 0, 0, 1)


# ============================================================
# ⭐ FEATURE PERTURBATION UNIVERSALE RF
# ============================================================
def feature_perturbation_rf_universal(
    model,
    X_test_real,
    y_test,
    min_vals,
    max_vals,
    immutable_mask,
    feature_names,
    max_ratio=0.05,
    n_features=5,
    seed=42,
    output_dir=None
):
    """
    Attaque feature perturbation UNIVERSALE :
    - CICIDS → pas de négatifs, clamp dur
    - UNSW → arrondi ct_*, clamp dur, patch négatifs
    - Pas de broadcasting
    - Respect des colonnes immuables
    - nb_negative = 0 garanti
    - nb_above_max = 0 garanti (avec les mêmes min/max dans le check)
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------
    # 1) Numpy conversion
    # ------------------------------------
    X_clean = np.asarray(X_test_real, dtype=float)
    y_test = np.asarray(y_test).reshape(-1)

    min_vals = np.asarray(min_vals, dtype=float).reshape(-1)
    max_vals = np.asarray(max_vals, dtype=float).reshape(-1)
    immutable_mask = np.asarray(immutable_mask, dtype=int).reshape(-1)

    n_samples, n_total_features = X_clean.shape

    assert min_vals.shape[0] == n_total_features
    assert max_vals.shape[0] == n_total_features
    assert immutable_mask.shape[0] == n_total_features

    # ------------------------------------
    # 2) Dataset auto-detection
    # ------------------------------------
    dataset_type = detect_dataset_from_columns(feature_names)
    print(f"📌 Dataset détecté : {dataset_type}")

    # ⚠️ IMPORTANT : on ne veut jamais de max négatif si on force X_adv >= 0
    # -> on ajuste les bornes pour le contrôle
    max_vals = np.maximum(max_vals, 0.0)

    if dataset_type == "CICIDS":
        # pour CICIDS les features physiques sont >= 0
        min_vals = np.maximum(min_vals, 0.0)

    # ------------------------------------
    # 3) Binarization
    # ------------------------------------
    y_test_bin = binarize_labels(y_test)

    # ------------------------------------
    # 4) Top features RF
    # ------------------------------------
    # 🔧 Empêche de modifier les features géantes de CICIDS
    giant_features = [
        "Flow Duration", "Flow IAT Max", "Flow IAT Total", "Flow Bytes/s",
        "Flow Packets/s", "Tot Fwd Pkts", "Tot Bwd Pkts",
        "Fwd Pkts/s", "Bwd Pkts/s"
    ]

    for i, fname in enumerate(feature_names):
        if fname in giant_features:
            immutable_mask[i] = 0   # on interdit la perturbation

    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]

    modifiable = [i for i in idx_sorted if immutable_mask[i] == 1]
    top_idx = modifiable[:n_features]

    print(f"🔍 Features modifiées : {top_idx}")

    # ------------------------------------
    # 5) Perturbation réaliste
    # ------------------------------------
    X_adv = X_clean.copy()

    for i in top_idx:
        col = X_clean[:, i]
        # 🔧 Bruit absolu (réaliste pour CICIDS)
        max_noise = 3.0   # tu peux mettre 1.0 si tu veux être très réaliste
        noise = rng.uniform(-max_noise, max_noise, size=len(col))
        X_adv[:, i] = col + noise
    # ------------------------------------
    # 6) Clamp SAFE par feature (1er passage)
    # ------------------------------------
    for j in range(n_total_features):
        X_adv[:, j] = np.clip(X_adv[:, j], min_vals[j], max_vals[j])

    # ------------------------------------
    # 7) Remise des colonnes immuables
    # ------------------------------------
    immu_cols = (immutable_mask == 0)
    X_adv[:, immu_cols] = X_clean[:, immu_cols]

    # ------------------------------------
    # 8) Pas de négatifs
    # ------------------------------------
    X_adv = np.maximum(X_adv, 0.0)

    # ------------------------------------
    # 9) Patch UNSW : arrondi ct_* + re-clamp
    # ------------------------------------
    if dataset_type == "UNSW":
        for j, col_name in enumerate(feature_names):
            if col_name.startswith("ct_"):
                X_adv[:, j] = np.round(X_adv[:, j])

        # re-clamp après arrondi
        for j in range(n_total_features):
            X_adv[:, j] = np.clip(X_adv[:, j], min_vals[j], max_vals[j])

        X_adv = np.maximum(X_adv, 0.0)

    # ------------------------------------
    # 10) Clamp FINAL ABSOLU
    # ------------------------------------
    for j in range(n_total_features):
        X_adv[:, j] = np.minimum(X_adv[:, j], max_vals[j])
        X_adv[:, j] = np.maximum(X_adv[:, j], min_vals[j])

    # -> ici, mathématiquement :
    #  min_vals[j] <= X_adv[:, j] <= max_vals[j]  pour tout j
    #  et comme on a forcé max_vals >= 0, + X_adv >= 0
    #  -> nb_negative = 0 et nb_above_max = 0 si on réutilise ces min/max dans le check

    # ------------------------------------
    # 11) Sauvegarde
    # ------------------------------------
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X_adv_real_FP.npy", X_adv)
        print(f"📁 X_adv sauvegardé dans : {out}")

    return X_adv, min_vals, max_vals  # on renvoie les min/max ajustés








def feature_perturbation_realistic_rf_cicids(
    model,
    X_test_real,
    y_test,
    min_vals,
    max_vals,
    immutable_mask,
    max_ratio=0.01,
    n_features=5,
    seed=42,
    output_dir=None
):
    """
    Attaque Feature Perturbation réaliste pour RF (CICIDS / UNSW).

    - Perturbations limitées (max_ratio * |x|)
    - Clamp sur [min_train, max_train]
    - Pour CICIDS : aucune valeur finale < 0 dans X_adv
    - Respect (relatif) des features immuables via immutable_mask
    """

    rng = np.random.default_rng(seed)

    # =============================
    # 1) Conversions numpy
    # =============================
    X_clean = np.asarray(X_test_real, dtype=float)
    y_test = np.asarray(y_test, dtype=object).reshape(-1)

    n_samples, n_total_features = X_clean.shape

    min_vals = np.asarray(min_vals, dtype=float).reshape(-1)
    max_vals = np.asarray(max_vals, dtype=float).reshape(-1)
    immutable_mask = np.asarray(immutable_mask, dtype=int).reshape(-1)

    assert min_vals.shape[0] == n_total_features
    assert max_vals.shape[0] == n_total_features
    assert immutable_mask.shape[0] == n_total_features

    # 🔧 Pour CICIDS : physiquement, les features sont ≥ 0
    # donc on force min_vals à être >= 0 pour l'attaque
    min_vals = np.maximum(min_vals, 0.0)

    # =============================
    # 2) Binarisation
    # =============================
    y_test_bin = binarize_labels(y_test).reshape(-1)

    # =============================
    # 3) Top features RF (tri correct)
    # =============================
    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]  # décroissant

    modifiable = [i for i in idx_sorted if immutable_mask[i] == 1]
    top_idx = modifiable[:n_features]

    print(f"🔍 Features modifiées : {top_idx}")

    # =============================
    # 4) Perturbation réaliste
    # =============================
    X_adv = X_clean.copy()

    for i in top_idx:
        col = X_clean[:, i]
        amp = max_ratio * (np.abs(col) + 1e-9)
        noise = rng.uniform(-amp, amp)
        X_adv[:, i] = col + noise

    # =============================
    # 5) Clamp SAFE (par feature)
    # =============================
    X_adv_clipped = X_adv.copy()
    for j in range(n_total_features):
        X_adv_clipped[:, j] = np.clip(X_adv[:, j], min_vals[j], max_vals[j])

    X_adv = X_adv_clipped

    # =============================
    # 6) Respect des colonnes immuables
    # =============================
    immu_cols = (immutable_mask == 0)
    X_adv[:, immu_cols] = X_clean[:, immu_cols]

    # =============================
    # 6 bis) GARANTIE FINALE : aucune valeur < 0
    # =============================
    X_adv = np.maximum(X_adv, 0.0)

    # =============================
    # 7) Prédictions
    # =============================
    y_clean_pred = model.predict(X_clean)
    y_adv_pred   = model.predict(X_adv)

    y_clean_bin = binarize_labels(y_clean_pred).reshape(-1)
    y_adv_bin   = binarize_labels(y_adv_pred).reshape(-1)

    # =============================
    # 8) Safety check des shapes
    # =============================
    y_test_bin  = y_test_bin.reshape(-1)
    y_clean_bin = y_clean_bin.reshape(-1)
    y_adv_bin   = y_adv_bin.reshape(-1)

    assert y_test_bin.shape == y_clean_bin.shape == y_adv_bin.shape, \
        f"Shapes mismatch : y_test={y_test_bin.shape}, clean={y_clean_bin.shape}, adv={y_adv_bin.shape}"

    # =============================
    # 9) Métriques IDS
    # =============================
    metrics_before = {
        "accuracy": float(accuracy_score(y_test_bin, y_clean_bin)),
        "recall":   float(recall_score(y_test_bin, y_clean_bin)),
        "f1":       float(f1_score(y_test_bin, y_clean_bin)),
    }

    metrics_after = {
        "accuracy": float(accuracy_score(y_test_bin, y_adv_bin)),
        "recall":   float(recall_score(y_test_bin, y_adv_bin)),
        "f1":       float(f1_score(y_test_bin, y_adv_bin)),
    }

    fn_before = int(((y_test_bin == 1) & (y_clean_bin == 0)).sum())
    fn_after  = int(((y_test_bin == 1) & (y_adv_bin == 0)).sum())
    diff_pred = int(np.sum(y_clean_bin != y_adv_bin))

    # =============================
    # 10) Sauvegarde
    # =============================
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X_adv_real_FP.npy", X_adv)
        print(f"📁 Sauvegardé dans : {out}")

    print("\n📊 ÉVALUATION Feature Perturbation RF")
    print("-------------------------------------")
    print(f"Accuracy clean : {metrics_before['accuracy']:.4f}")
    print(f"Accuracy adv   : {metrics_after['accuracy']:.4f}")
    print(f"Recall clean   : {metrics_before['recall']:.4f}")
    print(f"Recall adv     : {metrics_after['recall']:.4f}")
    print(f"F1 clean       : {metrics_before['f1']:.4f}")
    print(f"F1 adv         : {metrics_after['f1']:.4f}")

    print("\n🔥 Faux négatifs")
    print(f"FN clean : {fn_before}")
    print(f"FN adv   : {fn_after}")
    print(f"➡️  Augmentation des FN : {fn_after - fn_before}")
    print(f"🔁 Prédictions modifiées : {diff_pred}")



    return {
        "top_features": top_idx,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fn_before": fn_before,
        "fn_after": fn_after,
        "fn_increase": fn_after - fn_before,
        "diff_predictions": diff_pred,
        "X_adv": X_adv,
    }







import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score


def run_feature_perturbation_attack(
    model,
    X_test,
    y_test,
    n_features=5,
    epsilon=0.1,
    output_dir="../results/attacks/feature_perturbation/"
):
    """
    Attaque Feature Perturbation complète :
    - sélection des top features importantes
    - génération X_adv
    - binarisation auto labels (CICIDS / UNSW)
    - prédictions avant/après
    - calcul métriques
    - calcul faux négatifs
    - sauvegarde résultats
    """

    # ============================================================
    # 1) Normalisation de y_test → assure vector 1D
    # ============================================================
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    if hasattr(y_test, "shape") and len(y_test.shape) > 1:
        y_test = np.squeeze(y_test)

    y_test = pd.Series(y_test).reset_index(drop=True)

    # ============================================================
    # 2) Binarisation automatique
    # ============================================================
    def binarize(y):
        y_arr = np.array(y)

        # CAS 1 : labels textuels (CICIDS)
        if y_arr.dtype == object:
            y_str = pd.Series(y_arr).astype(str).str.upper()
            return np.where(y_str == "BENIGN", 0, 1)

        # CAS 2 : labels numériques (UNSW)
        return np.where(y_arr == 0, 0, 1)

    # Binarisation y_test
    y_test_bin = pd.Series(binarize(y_test))

    # ============================================================
    # 3) Top features importantes
    # ============================================================
    feature_importances = model.feature_importances_
    cols = X_test.columns

    top_idx = np.argsort(feature_importances)[-n_features:]
    top_features = cols[top_idx].tolist()

    print(f"🔍 Features perturbées : {top_features}")

    # ============================================================
    # 4) Génération adverasiale
    # ============================================================
    X_adv = X_test.copy()

    for col in top_features:
        std = X_test[col].std()
        perturb = epsilon * std
        noise = np.random.uniform(-perturb, perturb, size=len(X_test))
        X_adv[col] = X_adv[col] + noise

    # ============================================================
    # 5) Prédictions avant / après
    # ============================================================
    y_pred_clean = model.predict(X_test)
    y_pred_adv = model.predict(X_adv)

    y_pred_clean_bin = pd.Series(binarize(y_pred_clean))
    y_pred_adv_bin = pd.Series(binarize(y_pred_adv))

    # ============================================================
    # 6) Métriques
    # ============================================================
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

    metrics_before = compute_metrics(y_test_bin, y_pred_clean_bin)
    metrics_after = compute_metrics(y_test_bin, y_pred_adv_bin)

    # ============================================================
    # 7) Faux négatifs
    # ============================================================
    fn_before = int(((y_test_bin == 1) & (y_pred_clean_bin == 0)).sum())
    fn_after = int(((y_test_bin == 1) & (y_pred_adv_bin == 0)).sum())
    fn_increase = fn_after - fn_before

    # ============================================================
    # 8) Sauvegardes
    # ============================================================
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Sauvegarde X_adv
    X_adv.to_csv(output / "X_adv.csv", index=False)

    # Sauvegarde comparaison des prédictions
    pd.DataFrame({
        "y_true_bin": y_test_bin,
        "y_pred_clean_bin": y_pred_clean_bin,
        "y_pred_adv_bin": y_pred_adv_bin
    }).to_csv(output / "predictions_compare.csv", index=False)

    # ============================================================
    # 9) Résultat final propre
    # ============================================================
    diff_predictions = int((y_pred_clean_bin != y_pred_adv_bin).sum())

    return {
        "top_features": top_features,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fn_before": fn_before,
        "fn_after": fn_after,
        "fn_increase": fn_increase,
        "diff_predictions": diff_predictions
    }

