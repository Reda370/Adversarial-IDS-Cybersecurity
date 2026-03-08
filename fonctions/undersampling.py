import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

def undersample_dataset(X, y, name, seed=42, min_size=50000):
    """
    Undersampling intelligent et contrôlé.

    - Si le dataset est TROP PETIT (< min_size lignes) :
         → Undersampling désactivé pour éviter la perte d'information.
    
    - Si dataset assez grand :
         → Undersampling complet (RandomUnderSampler).

    Retour :
        X_resampled, y_resampled
    """

    print(f"\n=== 🔽 UNDERSAMPLING pour {name.upper()} ===")

    dataset_size = len(X)
    print(f"📏 Taille du dataset : {dataset_size} lignes")

    out_dir = Path(f"../data/undersampling/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CAS 1 : dataset trop petit → pas d'undersampling
    if dataset_size < min_size:
        print(f"❗ Dataset < {min_size} → undersampling désactivé.")
        print("➡ Retour du dataset original.")

        # Sauvegarde
        X.to_csv(out_dir / "X_train_under.csv", index=False)
        y.to_csv(out_dir / "y_train_under.csv", index=False)

        return X, y

    # CAS 2 : dataset assez grand → undersampling
    print("🟢 Dataset suffisamment grand → undersampling en cours...")

    rus = RandomUnderSampler(random_state=seed)   # <-- LINE CORRIGÉE
    X_under, y_under = rus.fit_resample(X, y)

    print(f"✔️ Terminé : {len(X_under)} lignes")

    # Sauvegarde
    X_under.to_csv(out_dir / "X_train_under.csv", index=False)
    y_under.to_csv(out_dir / "y_train_under.csv", index=False)

    return X_under, y_under
