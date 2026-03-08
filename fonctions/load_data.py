import pandas as pd
from pathlib import Path

def load_cicids(path):
    raw_dir = Path(path)
    csv_files = sorted(raw_dir.glob("*.csv"))
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)

def load_unsw(path):
    raw_dir = Path(path)
    train_df = pd.read_parquet(raw_dir / "UNSW_NB15_training-set.parquet")
    test_df  = pd.read_parquet(raw_dir / "UNSW_NB15_testing-set.parquet")
    return pd.concat([train_df, test_df], ignore_index=True)
