from pathlib import Path
from data import df
import pandas as pd

RANDOM_SEEDS = [42, 43, 44]
N_SPLITS = 5
FOLD_PATH = Path("data/splits/en-nn_folds_v1.csv")


def load_folds():
    """Đọc bảng gán compound/câu vào fold."""
    return pd.read_csv(FOLD_PATH)


def iter_folds(df, repeat):
    """Yield (fold, train_idx, val_idx) cho 5 fold của một lần lặp."""
    if repeat not in range(len(RANDOM_SEEDS)):
        raise ValueError(f"repeat phải nằm trong [0, {len(RANDOM_SEEDS) - 1}]")

    seed = RANDOM_SEEDS[repeat]
    manifest = load_folds()
    fold_col = f"fold_r{repeat}"

    # Sắp manifest theo thứ tự các câu trong df để index dùng được với df.iloc.
    manifest = manifest.set_index("ContextID").reindex(df["ContextID"])

    for fold in range(N_SPLITS):
        val_idx = (manifest[fold_col].to_numpy() == fold).nonzero()[0]
        train_idx = (manifest[fold_col].to_numpy() != fold).nonzero()[0]
        yield seed, fold, train_idx, val_idx

def main():
    # for i in range(0, 3):
    #     for seed, fold, train_idx, val_idx in iter_folds(df, repeat=i):
    #             train_df = df.iloc[train_idx]
    #             val_df = df.iloc[val_idx]
    #             print(
    #                 f"seed={seed}, fold={fold}, "
    #                 f"train={len(train_df)} câu / {train_df['Compound'].nunique()} compound, "
    #                 f"valid={len(val_df)} câu / {val_df['Compound'].nunique()} compound"
    #             )
    for seed, fold, train_idx, val_idx in iter_folds(df, repeat=2):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(
            f"seed={seed}, fold={fold}, "
            f"train={len(train_df)} câu / {train_df['Compound'].nunique()} compound, "
            f"valid={len(val_df)} câu / {val_df['Compound'].nunique()} compound"
        )

if __name__ == "__main__":
    main()