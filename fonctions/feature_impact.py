import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import accuracy_score

# ==========================================================
# 0) Fonction utilitaire : créer dossier automatiquement
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ==========================================================
# 1) Fonction principale : Compute Feature Impact (US17)
#    + enregistrement CSV + JSON
# ==========================================================
def compute_feature_impact(model, X_test, y_test, X_adv, feature_names, save_dir):
    """
    Calcule l’impact réel par feature (US17) :
    - Perturbe UNE seule feature à la fois
    - Mesure la chute d’accuracy
    - Sauvegarde :
        • feature_impact.csv
        • top20_features.json
    """
    
    ensure_dir(save_dir)

    # Accuracy baseline
    y_pred_before = model.predict(X_test)
    acc_before = accuracy_score(y_test, y_pred_before)

    print(f"\nAccuracy avant attaque : {acc_before:.4f}")

    impact_dict = {}

    # Calcul impact par feature
    for i, name in enumerate(feature_names):

        X_temp = X_test.copy()
        X_temp[:, i] = X_adv[:, i]  # Perturbe seulement la feature i

        y_pred_i = model.predict(X_temp)
        acc_i = accuracy_score(y_test, y_pred_i)

        drop_i = acc_before - acc_i

        impact_dict[name] = {
            "acc_original": acc_before,
            "acc_with_feature_perturbed": acc_i,
            "drop": drop_i
        }

    # Tri décroissant
    sorted_features = sorted(
        impact_dict.items(),
        key=lambda x: x[1]["drop"],
        reverse=True
    )

    # Sauvegarde CSV
    df = pd.DataFrame({
        "feature": [f for f, _ in sorted_features],
        "drop": [v["drop"] for _, v in sorted_features]
    })

    csv_path = os.path.join(save_dir, "feature_impact.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 feature_impact.csv sauvegardé dans {csv_path}")

    # Sauvegarde JSON top 20
    json_path = os.path.join(save_dir, "top20_features.json")
    with open(json_path, "w") as f:
        json.dump(sorted_features[:20], f, indent=4)
    print(f"💾 top20_features.json sauvegardé dans {json_path}")

    return impact_dict, sorted_features


# ==========================================================
# 2) Heatmap d’impact + sauvegarde PNG
# ==========================================================
def plot_feature_impact_heatmap(sorted_features, save_dir, top_k=20):
    """
    Crée et sauvegarde une heatmap des top-K features les plus sensibles.
    """

    ensure_dir(save_dir)

    top = sorted_features[:top_k]

    df = pd.DataFrame({
        "Feature": [f for f,_ in top],
        "Drop": [v["drop"] for _,v in top]
    }).set_index("Feature")

    plt.figure(figsize=(10, max(4, top_k * 0.4)))
    sns.heatmap(df, annot=True, cmap="Reds", linewidths=0.5)
    plt.title(f"Feature Sensitivity Heatmap (Top {top_k})")

    save_path = os.path.join(save_dir, "heatmap_top20.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"💾 heatmap_top20.png sauvegardée dans {save_path}")


# ==========================================================
# 3) Boxplot Before/After Feature + sauvegarde PNG
# ==========================================================
def plot_feature_boxplot(feature_name, feature_index, X_test, X_adv, save_dir):
    """
    Crée et sauvegarde un boxplot Before/After pour une feature.
    """

    ensure_dir(save_dir)

    data = [X_test[:, feature_index], X_adv[:, feature_index]]

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=data)
    plt.xticks([0, 1], ["Before Attack", "After Attack"])
    plt.title(f"Distribution Before/After - {feature_name}")
    plt.xlabel("Dataset")
    plt.ylabel("Feature Value")

    file_name = f"boxplot_{feature_name.replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"💾 boxplot sauvegardé : {save_path}")
