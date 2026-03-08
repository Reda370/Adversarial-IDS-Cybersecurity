import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance(model, X, top_k=20, title="Feature Importance", save_path="results"):
    """
    Affiche ET sauvegarde les top_k features les plus importantes d'un modèle.
    - Affichage direct dans Jupyter
    - Sauvegarde automatique dans le dossier 'results/'
    """

    print("\n📊 Extraction des features importantes...")

    # Importance brute
    importance = model.feature_importances_

    # DataFrame
    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False).head(top_k)

    # --- Création du dossier results s'il n'existe pas ---
    os.makedirs(save_path, exist_ok=True)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        y="feature",
        x="importance",
        color="steelblue"   # une seule couleur → pas de warning seaborn
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    # --- Chemin complet du fichier ---
    file_path = os.path.join(save_path, f"{title.replace(' ','_').lower()}.png")

    # --- Sauvegarde ---
    plt.savefig(file_path, dpi=300)
    print(f"📁 Graphique sauvegardé dans : {file_path}")

    # --- Affichage ---
    plt.show()

    return df
