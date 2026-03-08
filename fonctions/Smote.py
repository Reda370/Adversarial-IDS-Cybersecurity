from imblearn.over_sampling import SMOTE
import pandas as pd
from pathlib import Path

def apply_smote(X_train, y_train, dataset_name="dataset", seed=42, limit=15000, k_fixed=5):
    """
    SMOTE Safe Mode :
    - Réduction fixe : limit lignes (SMOTE ne supporte pas plus)
    - k_neighbors fixé (5 par défaut) pour un SMOTE réaliste et stable
    - Compatible UNSW / CICIDS
    """

    print("\n🔵 === SMOTE Safe Mode ===\n")

    
    # Normalisation
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    
    label_col = y_train.name
    df = pd.concat([X_train, y_train], axis=1)

    n = len(df)
    print(f"📏 Taille initiale : {n} lignes")

    
    # Réduction fixe
    if n > limit:
        print(f"⚠️ Réduction à {limit} lignes pour compatibilité SMOTE")
        df = df.sample(limit, random_state=seed)

    print(f"📏 Taille finale : {len(df)} lignes")

    
    # Suppression classes rares
    class_counts = df[label_col].value_counts()
    rare = class_counts[class_counts < 3].index.tolist()
    
    if rare:
        print(f"⚠️ Classes rares supprimées : {rare}")
        df = df[~df[label_col].isin(rare)]

    X = df.drop(columns=label_col)
    y = df[label_col]

    
    # 🔧 k_neighbors FIXE mais safe
    min_count = y.value_counts().min()
    k = min(k_fixed, min_count - 1)
    k = max(1, k)

    print(f"🔧 k_neighbors ajusté : {k}")

    print("\n🟢 Application SMOTE…")
    sm = SMOTE(k_neighbors=k, random_state=seed)
    X_smote, y_smote = sm.fit_resample(X, y)

    print("\n📊 Distribution après SMOTE :")
    print(pd.Series(y_smote).value_counts())

    
    # Sauvegarde
    save_dir = Path(f"../data/smote/{dataset_name}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_smote).to_csv(save_dir / "X_train_smote.csv", index=False)
    pd.DataFrame({label_col: y_smote}).to_csv(save_dir / "y_train_smote.csv", index=False)

    print("\n💾 Données SMOTE sauvegardées.")

    
    return X_smote, y_smote
