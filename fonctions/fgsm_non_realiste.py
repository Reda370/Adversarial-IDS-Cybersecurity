import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score


def run_fgsm_attack_mlp(
    model,
    X_test_np,
    y_test_np,
    epsilon=0.05,
    batch_size=1024,
    output_dir ="../attacks/fgsm",
    seed=42
):
    """
    Attaque FGSM sur MLP avec device AUTO :
    - Utilise CUDA si disponible, sinon CPU
    - Complètement reproductible (seed fixe)
    """


    # ============================================================
    # 0) Device AUTO (cuda si dispo, sinon cpu)
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Device utilisé pour FGSM : {device}")

    # Seed = reproductible
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Envoyer modèle sur device
    model = model.to(device)
    model.eval()

    # Données → tensors device
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # 1) Prédictions propres
    # ============================================================
    def predict_batches(X):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                logits = model(batch)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds)

    y_pred_clean = predict_batches(X_test)

    # ============================================================
    # 2) FGSM
    # ============================================================
    X_adv = X_test.clone().detach().requires_grad_(True)

    logits = model(X_adv)
    loss = criterion(logits, y_test)

    model.zero_grad()
    loss.backward()

    grad_sign = X_adv.grad.data.sign()

    # FGSM
    X_adv = X_adv + epsilon * grad_sign

    # Clamp dans les min/max du test
    X_min = X_test.min(0, keepdim=True).values
    X_max = X_test.max(0, keepdim=True).values
    X_adv = torch.max(torch.min(X_adv, X_max), X_min)

    X_adv = X_adv.detach()

    # ============================================================
    # 3) Prédictions adversariales
    # ============================================================
    y_pred_adv = predict_batches(X_adv)

    # ============================================================
    # 4) Métriques
    # ============================================================
    y_true = y_test.cpu().numpy()

    def metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

    metrics_before = metrics(y_true, y_pred_clean)
    metrics_after  = metrics(y_true, y_pred_adv)

    fn_before = int(((y_true == 1) & (y_pred_clean == 0)).sum())
    fn_after  = int(((y_true == 1) & (y_pred_adv == 0)).sum())

    diff_predictions = int((y_pred_clean != y_pred_adv).sum())

    # ============================================================
    # 5) Sauvegarde
    # ============================================================
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "y_true": y_true,
        "y_pred_clean": y_pred_clean,
        "y_pred_adv": y_pred_adv
    }).to_csv(output / "predictions_fgsm.csv", index=False)

    np.save(output / "X_adv_sample.npy", X_adv[:100].cpu().numpy())
    # === Sauvegarde du dataset complet X_adv ===
    np.save(output / "X_adv_full.npy", X_adv.cpu().numpy())



    # ============================================================
    # 6) Résultat final
    # ============================================================
    return {
        "epsilon": epsilon,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fn_before": fn_before,
        "fn_after": fn_after,
        "fn_increase": fn_after - fn_before,
        "diff_predictions": diff_predictions,
        "device_used": device
    }
