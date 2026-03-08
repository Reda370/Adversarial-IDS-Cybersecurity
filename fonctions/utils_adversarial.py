def save_adversarial_dataset(
        X_adv_path,
        X_test,
        y_test,
        attack_name,
        model_name,
        dataset_name,
        balancing,
        plausibility_path,
        epsilon=None,
        base_dir="../data/adversarial"
):
    import os
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime

    # ============================================================
    # 1. Charger X_adv (CSV ou NPY)
    # ============================================================
    if X_adv_path.endswith(".csv"):
        X_adv = pd.read_csv(X_adv_path)
        print(f"📥 Loaded X_adv (CSV) from {X_adv_path}")

    elif X_adv_path.endswith(".npy"):
        X_adv_np = np.load(X_adv_path)
        X_adv = pd.DataFrame(X_adv_np)
        print(f"📥 Loaded X_adv (NPY) from {X_adv_path}")

    else:
        raise ValueError("❌ X_adv_path must be .csv or .npy")


    # ============================================================
    # 2. Convertir X_test & y_test correctement
    # ============================================================
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)


    # ============================================================
    # 3. Charger les stats de plausibilité
    # ============================================================
    with open(plausibility_path) as f:
        plausibility_stats = json.load(f)


    # ============================================================
    # 4. Calcul du delta automatiquement
    # ============================================================
    delta = X_adv.values - X_test.values


    # ============================================================
    # 5. Construire le chemin final
    # ============================================================
    dataset_dir = os.path.join(
        base_dir,
        dataset_name,
        attack_name,
        model_name,
        balancing
    )

    if epsilon is not None:
        eps_dir = f"eps_{str(epsilon).replace('.', '_')}"
        run_dir = os.path.join(dataset_dir, eps_dir)
    else:
        run_dir = dataset_dir


    # ============================================================
    # 6. Versioning automatique
    # ============================================================
    final_dir = run_dir
    version = 1
    while os.path.exists(final_dir):
        final_dir = f"{run_dir}_v{version}"
        version += 1

    os.makedirs(final_dir, exist_ok=True)


    # ============================================================
    # 7. Sauvegarde des données
    # ============================================================
    X_adv.to_csv(os.path.join(final_dir, "X_adv.csv"), index=False)
    y_test.to_csv(os.path.join(final_dir, "y_true.csv"), index=False)
    pd.DataFrame(delta).to_csv(os.path.join(final_dir, "delta.csv"), index=False)


    # ============================================================
    # 8. Sauvegarde des métadonnées
    # ============================================================
    metadata = {
        "dataset": dataset_name,
        "attack": attack_name,
        "model_used": model_name,
        "balancing": balancing,
        "epsilon": epsilon,
        "nb_samples": len(X_adv),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plausibility_stats": plausibility_stats
    }

    with open(os.path.join(final_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Dataset adversarial sauvegardé dans : {final_dir}")

    return final_dir
