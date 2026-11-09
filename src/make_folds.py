from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
from data import df

RANDOM_SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BINS = 3

def build_strata(df, n_bins=N_BINS):
    type_mod_mean = df.groupby("Compound")["ModAvg"].mean()
    type_head_mean = df.groupby("Compound")["HeadAvg"].mean()
    
    bins_mod, edges_mod = pd.qcut(
        type_mod_mean,
        q=3,
        labels=False,
        duplicates="drop",
        retbins=True
    )
    bins_head, edges_head = pd.qcut(
        type_head_mean,
        q=3,
        labels=False,
        duplicates="drop",
        retbins=True
    )
    # print("Các ranh giới Bin:", edges)
    # chia compound dựa trên phân vị của ModAvg
    # Các ranh giới Bin: [0.31428571  3.72      4.32       4.871112  ]
    # ví dụ nhóm "zinc plate" có mean ModAvg là 3.9 sẽ được gán vào bin 1
    # ví dụ nhóm "zinc plate" có mean HeadAvg là 1.5 sẽ được gán vào bin 0
    # ví dụ nhóm "zinc plate" có cặp (bin_mod, bin_head) = (1, 0) sẽ được gán vào strata 1_0
    # gán số bin ở trên cho từng compound trong nhóm đó

    strata = bins_mod.astype(str) + "_" + bins_head.astype(str)
    return df["Compound"].map(strata)


def make_group_folds(data, seeds=RANDOM_SEEDS, n_splits=N_SPLITS):
    data = data.reset_index(drop=True).copy()
    strata = build_strata(data)

    all_folds = {}
    manifest = data[["ContextID", "Compound"]].copy()
    manifest["stratum"] = strata

    for r, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        seed_folds = []
        fold_col = f"fold_r{r}"
        split_col = f"split_r{r}"

        manifest[fold_col] = -1
        manifest[split_col] = ""

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(data, y=strata, groups=data["Compound"])
        ):
            train_groups = set(data.iloc[train_idx]["Compound"])
            val_groups = set(data.iloc[val_idx]["Compound"])

            assert train_groups.isdisjoint(val_groups)
            
            manifest.loc[val_idx, fold_col] = fold

            # Gán train / validation
            manifest.loc[train_idx, split_col] = "train"
            manifest.loc[val_idx, split_col] = "validation"

            seed_folds.append((train_idx, val_idx))

        all_folds[seed] = seed_folds # mỗi seed lưu 5 cặp (train_idx, val_idx) tương ứng với 5 fold

    return all_folds, manifest

def main():
    folds, manifest = make_group_folds(df)

    manifest.to_csv("./data/splits/en-nn_folds_v1.csv", index=False)

    ACTIVE_SEED = 42 # key
    ACTIVE_FOLD = 0 # value, ứng với fold 0 trong 5 fold

    train_idx, val_idx = folds[ACTIVE_SEED][ACTIVE_FOLD]

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    print("Train:", len(train_df), train_df["Compound"].nunique())
    print("Valid:", len(val_df), val_df["Compound"].nunique())

    print(
        "Overlap:",
        len(set(train_df.Compound) & set(val_df.Compound))
    )

if __name__ == "__main__":
    main()