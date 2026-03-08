import numpy as np
import pandas as pd
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from .feature_attack import feature_perturbation_rf_universal
from .plausibility import check_plausibility_RF
from .model_rf import load_model
from .fgsm import get_immutable_mask_for_cicids_final, get_immutable_mask_unsw


def defense_feature_perturbation_rf(
    X_train_path,
    y_train_path,
    baseline_model_path,
    save_model_path,
    dataset="CICIDS",     # "CICIDS" ou "UNSW"
    max_ratio=0.05,
    n_features=5,
    seed=42
):
    """
    Défense RF complète :
    - charge X_train
    - masque auto
    - attaque FP sur TRAIN
    - plausibility TRAIN
    - concat X_train + X_adv
    - entraîne modèle défendu
    - sauvegarde modèle
    """

    print(f"\n=== DEFENSE FEATURE PERTURBATION RF ({dataset}) ===")

    # 1. Charger train
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]

    # 2. Charger baseline
    model = load_model(baseline_model_path)

    # 3. Masque immuable auto
    if dataset == "CICIDS":
        immutable_mask = get_immutable_mask_for_cicids_final(X_train.columns)
    else:
        immutable_mask = get_immutable_mask_unsw(X_train.columns)

    # 4. Min/max train
    min_vals = X_train.values.min(axis=0)
    max_vals = X_train.values.max(axis=0)

    # 5. Attaque FP sur TRAIN
    X_adv_train, min_eff, max_eff = feature_perturbation_rf_universal(
        model=model,
        X_test_real=X_train.values,
        y_test=y_train.values,
        min_vals=min_vals,
        max_vals=max_vals,
        immutable_mask=immutable_mask,
        feature_names=X_train.columns,
        max_ratio=max_ratio,
        n_features=n_features,
        seed=seed,
        output_dir=os.path.join(os.path.dirname(save_model_path), "FP_attack_train")
    )

    # 6. Plausibility TRAIN
    stats, verdict = check_plausibility_RF(
        X_clean=X_train.values,
        X_adv=X_adv_train,
        min_vals=min_eff,
        max_vals=max_eff,
        output_dir=os.path.join(os.path.dirname(save_model_path), "FP_attack_train")
    )

    print("Plausibility verdict :", verdict)

    # 7. Dataset défendu
    X_def = np.vstack([X_train.values, X_adv_train])
    y_def = np.concatenate([y_train.values, y_train.values])

    # 8. Modèle défendu
    rf_def = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf_def.fit(X_def, y_def)

    # 9. Sauvegarde
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    dump(rf_def, save_model_path)

    print(f"✔ Modèle défendu sauvegardé dans : {save_model_path}")

    return rf_def
