import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path

def oversample_dataset(X, y, name, seed=42, max_size=250000):
    """
    Oversampling intelligent et contrôlé.
    
    - Si le dataset est PETIT (moins de max_size lignes) :
         → Oversampling complet (RandomOverSampler).
    
    - Si le dataset est GRAND (plus de max_size lignes) :
         → Aucun oversampling (protection mémoire).
    
    Paramètres :
        X, y : features et labels
        name : nom du dataset ("cicids", "unsw", etc.)
        seed : reproductibilité
        max_size : seuil à partir duquel on désactive l'oversampling
    
    Retour :
        X_resampled, y_resampled
    """

    print(f"\n=== 🔁 Oversampling pour {name.upper()} ===")

    # Taille du dataset actuel
    dataset_size = len(X)
    print(f"📏 Taille du dataset : {dataset_size} lignes")

    # =====================================================
    # 1️⃣ CAS : dataset trop grand → oversampling désactivé
    # =====================================================
    if dataset_size > max_size:
        print(f"❗ Dataset > {max_size} lignes → oversampling désactivé.")
        print("➡ Retour du dataset original sans modification.")

        # Sauvegarde "as is"
        out_dir = Path(f"../data/oversampling/{name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        X.to_csv(out_dir / "X_train_over.csv", index=False)
        y.to_csv(out_dir / "y_train_over.csv", index=False)

        print(f"📁 Données sauvegardées sans oversampling dans : {out_dir}")
        return X, y

    # =====================================================
    # 2️⃣ CAS : dataset petit → oversampling complet
    # =====================================================
    print("🟢 Dataset avec taille raisonable → oversampling complet en cours...")

    ros = RandomOverSampler(random_state=seed)
    X_over, y_over = ros.fit_resample(X, y)

    print(f"✔️ Oversampling terminé : {len(X_over)} lignes")

    # Sauvegarde du dataset équilibré
    out_dir = Path(f"../data/oversampling/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    X_over.to_csv(out_dir / "X_train_over.csv", index=False)
    y_over.to_csv(out_dir / "y_train_over.csv", index=False)

    print(f"📁 Données équilibrées sauvegardées dans : {out_dir}")

    return X_over, y_over
