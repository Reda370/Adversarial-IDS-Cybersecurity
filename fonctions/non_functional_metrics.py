# ============================================================
# non_functional_metrics.py
# Mesures non fonctionnelles : latence, mémoire + extraction modèle
# ============================================================

import time
import os
import json
import numpy as np
import psutil
import torch



import joblib




# ------------------------------------------------------------
# EXTRACTION AUTOMATIQUE DU MODÈLE (clé fixée ici)
# ------------------------------------------------------------
def extract_model(obj):
    """
    Rend un modèle compatible .predict() ou un modèle PyTorch.
    Gère les cas :
    - modèle direct
    - tuple (model, stats)
    - dict {"model": model}
    """

    # Si c'est déjà un modèle sklearn
    if hasattr(obj, "predict"):
        return obj

    # Si c'est un tuple -> prendre premier élément
    if isinstance(obj, tuple) and len(obj) > 0:
        first = obj[0]
        if hasattr(first, "predict"):
            return first

    # Si c'est un dictionnaire -> prendre la clé "model"
    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"]

    raise ValueError("❌ Impossible d'extraire un modèle valide depuis l'objet chargé.")


# ------------------------------------------------------------
# Mesure temps d'inférence sklearn
# ------------------------------------------------------------
def measure_latency_sklearn(model, X_sample, n_runs=200):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_sample.reshape(1, -1))
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


# ------------------------------------------------------------
# Mesure temps d'inférence MLP PyTorch
# ------------------------------------------------------------
def measure_latency_mlp(model, X_sample, device="cpu", n_runs=200):
    model.eval()
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(X_tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


# ------------------------------------------------------------
# Taille modèle sur disque
# ------------------------------------------------------------
def measure_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    return round(size_bytes / (1024 * 1024), 4)


# ------------------------------------------------------------
# Mémoire utilisée
# ------------------------------------------------------------
def measure_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return round(mem_bytes / (1024 * 1024), 4)


# ------------------------------------------------------------
# Fonction PRINCIPALE
# ------------------------------------------------------------
def evaluate_non_functional(model, model_path, X_test, is_mlp=False, device="cpu"):

    # 🔥 Correction ic
