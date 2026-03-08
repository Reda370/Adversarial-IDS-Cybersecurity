import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_anomaly_detector(
        X_train: np.ndarray,
        contamination: float = 0.01,
        random_state: int = 42
    ):
    """
    Entraîne un IsolationForest sur données propres (US21).
    """
    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )
    detector.fit(X_train)
    return detector


def detect_samples(detector, X: np.ndarray):
    """
    Retourne un vecteur binaire :
        1 = anomalie détectée
        0 = normal
    """
    return (detector.predict(X) == -1).astype(int)


def evaluate_detector(
        detector,
        X_clean: np.ndarray,
        X_adv: np.ndarray
    ):
    """
    Calcul FA% et DR% (métriques US21).
    """
    y_clean = detect_samples(detector, X_clean)
    y_adv = detect_samples(detector, X_adv)

    return {
        "FA_percent": float(y_clean.mean() * 100),
        "DR_percent": float(y_adv.mean() * 100),
        "nb_clean": int(len(X_clean)),
        "nb_adv": int(len(X_adv))
    }


def save_us21_results(metrics: dict, output_dir: str):
    """
    Sauvegarde du fichier JSON des résultats de l'US21.
    """
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "US21_results.json")

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📁 Résultats sauvegardés dans : {save_path}")

    return save_path


def run_US21(
        X_train: np.ndarray,
        X_clean: np.ndarray,
        X_adv: np.ndarray,
        output_dir: str
    ):
    """
    Pipeline complet US21 :
    - train IsolationForest
    - evaluate FA% / DR%
    - save JSON
    - return metrics + detector
    """
    detector = train_anomaly_detector(X_train)
    metrics = evaluate_detector(detector, X_clean, X_adv)
    save_us21_results(metrics, output_dir)

    return detector, metrics
