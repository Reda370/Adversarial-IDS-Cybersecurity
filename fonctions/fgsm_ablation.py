import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from .fgsm import run_fgsm_finale


def run_fgsm_ablation(
        model,
        X_test_np,
        y_test_np,
        scaler,
        min_vals,
        max_vals,
        immutable_mask,
        save_dir,
        epsilons=None
    ):
    """
    US18 – Ablation FGSM réaliste
    ❗ Version correcte : uniquement pour MLP PyTorch
    """

    # epsilons officiels pour projet étudiant cyber
    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("\n🚀 US18 – Début de l'expérience d'ablation FGSM (MLP uniquement)")

    # ============================================================
    # 1) METRICS CLEAN (prédictions MLP)
    # ============================================================
    X_clean_norm = scaler.transform(X_test_np)
    preds_clean = model(torch.tensor(X_clean_norm, dtype=torch.float32)).argmax(1).cpu().numpy()

    f1_clean  = f1_score(y_test_np, preds_clean)
    acc_clean = accuracy_score(y_test_np, preds_clean)
    rec_clean = recall_score(y_test_np, preds_clean)
    prec_clean = precision_score(y_test_np, preds_clean)
    fn_clean = int(((y_test_np == 1) & (preds_clean == 0)).sum())

    results = []

    # ============================================================
    # 2) BOUCLE ABLATION FGSM
    # ============================================================
    for eps in epsilons:
        print(f"\n⚡ FGSM epsilon = {eps}")

        # ATTENTION : on attaque le MLP
        X_adv_real, X_adv_norm, preds_adv = run_fgsm_finale(
            model=model,
            X_test_np=X_test_np,
            y_test_np=y_test_np,
            scaler=scaler,
            min_vals=min_vals,
            max_vals=max_vals,
            immutable_mask=immutable_mask,
            epsilon=eps,
            max_ratio=0.05,
            output_dir=f"{save_dir}/eps_{eps}"
        )

        # prédictions adversariales (toujours MLP)
        acc_adv = accuracy_score(y_test_np, preds_adv)
        rec_adv = recall_score(y_test_np, preds_adv)
        f1_adv  = f1_score(y_test_np, preds_adv)
        prec_adv = precision_score(y_test_np, preds_adv)
        fn_adv = int(((y_test_np == 1) & (preds_adv == 0)).sum())

        results.append({
            "epsilon": eps,
            "f1_clean": f1_clean,
            "f1_adv": f1_adv,
            "acc_clean": acc_clean,
            "acc_adv": acc_adv,
            "rec_clean": rec_clean,
            "rec_adv": rec_adv,
            "prec_clean": prec_clean,
            "prec_adv": prec_adv,
            "fn_clean": fn_clean,
            "fn_adv": fn_adv,
            "fn_increase": fn_adv - fn_clean
        })

    # ============================================================
    # 3) CSV + TXT
    # ============================================================
    df = pd.DataFrame(results)

    csv_path = f"{save_dir}/ablation_results.csv"
    df.to_csv(csv_path, index=False)

    txt_path = f"{save_dir}/ablation_results.txt"
    with open(txt_path, "w") as f:
        f.write(df.to_string(index=False))

    print(f"\n📁 CSV sauvegardé : {csv_path}")
    print(f"📄 TXT sauvegardé : {txt_path}")

    return df
