import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# ============================================================
# ⭐ FONCTION FINALE : evaluate_model
# ============================================================
def evaluate_model(model, X, y, output_dir):
    """
    Évalue un modèle (sklearn OU PyTorch).
    - Calcule accuracy, precision, recall, f1
    - Sauvegarde metrics.csv
    - Génère classification_report (.txt + .csv)
    - Génère confusion_matrix.png
    - Génère feature importance si sklearn RF
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("🤖 Prédiction du modèle...")

    # ======================================
    # 🔍 1) Prédiction : sklearn vs PyTorch
    # ======================================
    if hasattr(model, "predict"):  
        # Cas sklearn
        y_pred = model.predict(X)

    else:
        # Cas PyTorch
        model.eval()
        with torch.no_grad():
            # Conversion X → tensor float32
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(X_tensor)
            y_pred = logits.argmax(dim=1).cpu().numpy()

    # ======================================
    # 2) Calcul des métriques
    # ======================================
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
    }

    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    print("📄 metrics.csv généré.")

    # ======================================
    # 3) Classification report
    # ======================================
    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).to_csv(output / "classification_report.csv")
    
    with open(output / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y, y_pred, zero_division=0))
    
    print("📄 classification_report.csv + .txt générés.")

    # ======================================
    # 4) Confusion Matrix
    # ======================================
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png")
    plt.close()

    print("🖼️ confusion_matrix.png générée.")

    # ======================================
    # 5) Feature importance (SKLEARN ONLY)
    # ======================================
    if hasattr(model, "feature_importances_"):
        try:
            fi_dir = output / "feature_importance"
            fi_dir.mkdir(exist_ok=True)

            df_fi = pd.DataFrame({
                "feature": X.columns if isinstance(X, pd.DataFrame) else range(len(model.feature_importances_)),
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            df_fi.to_csv(fi_dir / "feature_importance.csv", index=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_fi.head(20), x="importance", y="feature", color="steelblue")
            plt.title("Top 20 Feature Importance")
            plt.tight_layout()
            plt.savefig(fi_dir / "feature_importance.png", dpi=300)
            plt.close()

            print("🖼️ feature_importance.png générée.")

        except Exception as e:
            print(f"⚠️ Impossible de générer la feature importance : {e}")

    print("🎉 Évaluation terminée !")

    return metrics, y_pred



# ============================================================
# 🔢 Fonction pour tracer la matrice de confusion
# ============================================================
def plot_confusion(y_test, y_pred):
    """
    Affiche la matrice de confusion sous forme de heatmap.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    plt.show()



# ============================================================
# 🔢 Fonction pour afficher les metrics
# ============================================================
def display_metrics(metrics):
    """
    Affichage clair des métriques dans le notebook.
    """
    print("\n📊 Metrics (Notebook Display)")
    print("-----------------------------")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro   : {metrics['recall_macro']:.4f}")
    print(f"F1 macro       : {metrics['f1_macro']:.4f}")
    print("\n📊 Metrics affichés")



# ============================================================
# 🔢 Fonction pour afficher le classification report
# ============================================================
def display_classification_report(y_true, y_pred):
    """
    Affiche le classification report complet dans le notebook.
    """
    print("\n📝 Classification Report")
    print("-----------------------------")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\n📝 Classification Report affiché")
