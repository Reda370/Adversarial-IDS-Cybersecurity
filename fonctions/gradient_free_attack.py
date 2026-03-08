import numpy as np
import warnings
from tqdm import tqdm
import json
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ============================================================
#  SAVE RESULTS
# ============================================================
def save_attack_results(X_adv, stats, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "X_adv.npy", X_adv)
    with open(save_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print(f"💾 Résultats sauvegardés dans : {save_dir}")


def decision_based_attack_rf_strong(
    model,
    X,
    y,
    min_vals,
    max_vals,
    immutable_mask,
    n_samples=None,
    max_iter=80,
    step_size=0.1,
    tries_per_iter=15,
    verbose=True,
    dataset="CICIDS"
):
    """
    Version 100% robuste :
    - Binarisation automatique des labels DU MODEL et de y
    - Compatible RF multiclass CICIDS (string labels)
    - Compatible RF binaire (int labels)
    """

    # ===============================================
    # 1️⃣ Conversion y_test → binaire
    # ===============================================
    y = np.array([0 if str(lbl).upper() == "BENIGN" else 1 for lbl in y], dtype=int)

    # ===============================================
    # 2️⃣ Conversion numpy
    # ===============================================
    X = np.asarray(X, float)
    min_vals = np.asarray(min_vals, float)
    max_vals = np.asarray(max_vals, float)
    immutable_mask = np.asarray(immutable_mask, bool)

    # ===============================================
    # 3️⃣ Sélection des samples malicieux
    # ===============================================
    malicious_idx = np.where(y == 1)[0]
    if n_samples is not None:
        malicious_idx = malicious_idx[:n_samples]

    X_adv = X.copy()
    success = 0
    fail = 0

    # ===============================================
    # 4️⃣ Fonction de conversion prédiction → binaire
    # ===============================================
    def to_binary(pred):
        """
        Convertit une prédiction du modèle en 0/1, qu'elle soit string ou int.
        """
        if isinstance(pred, str):
            return 0 if pred.upper() == "BENIGN" else 1
        return int(pred)

    # ===============================================
    # 5️⃣ Boucle principale d’attaque
    # ===============================================
    for i in tqdm(malicious_idx, desc="🔥 Robust Decision-Based Attack", ncols=90):

        x_adv = X_adv[i].copy()

        pred0_raw = model.predict(x_adv.reshape(1, -1))[0]
        pred0 = to_binary(pred0_raw)

        if pred0 == 0:
            continue  # déjà BENIGN

        flipped = False

        for it in range(max_iter):

            best_candidate = None
            best_score = 1.0

            for _ in range(tries_per_iter):

                noise = np.random.uniform(-step_size, step_size, size=x_adv.shape)
                noise[immutable_mask] = 0.0

                x_cand = np.clip(x_adv + noise, min_vals, max_vals)

                pred_raw = model.predict(x_cand.reshape(1, -1))[0]
                pred = to_binary(pred_raw)

                # ---- flip trouvé ----
                if pred == 0:
                    X_adv[i] = x_cand
                    success += 1
                    flipped = True
                    break

                # ---- score ----
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x_cand.reshape(1, -1))[0]
                    score = float(proba[1])  # proba attaque
                else:
                    score = float(pred)

                if score < best_score:
                    best_score = score
                    best_candidate = x_cand

            if flipped:
                break

            if best_candidate is not None:
                x_adv = best_candidate

        if not flipped:
            fail += 1

    stats = {
        "success": success,
        "fail": fail,
        "success_rate": success / max(1, success + fail),
        "attacked_samples": len(malicious_idx),
        "dataset": dataset
    }

    if verbose:
        print("\n===== 🔥 ROBUST DECISION-BASED ATTACK FINISHED =====")
        print(stats)

    return X_adv, stats
