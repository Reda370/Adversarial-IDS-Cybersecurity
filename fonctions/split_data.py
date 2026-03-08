import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from pathlib import Path

def split_dataset(X, y, name, seed=42):
    """
    Split un dataset en train (70%), validation (15%) et test (15%),
    avec stratification + seed fixe.

    Paramètres :
        X : DataFrame des features prétraitées
        y : Series du label
        name : str ("cicids" ou "unsw")
        seed : int
    """

    # Fixation de la seed
    np.random.seed(seed)
    random.seed(seed)

    # 70% train / 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=seed
    )

    # 30% → 15% val + 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed
    )

    # Dossier de sauvegarde
    out_dir = Path(f"../data/splits/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegardes
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    X_val.to_csv(out_dir / "X_val.csv", index=False)
    y_val.to_csv(out_dir / "y_val.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    print(f"✔️ Split terminé et sauvegardé dans : {out_dir}")

    return X_train, X_val, X_test, y_train, y_val, y_test
