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
    q_mod = min(n_bins, type_mod_mean.nunique())
    q_head = min(n_bins, type_head_mean.nunique())
    bins_mod, edges_mod = pd.qcut(
        type_mod_mean,
        q=q_mod,
        labels=False,
        duplicates="drop",
        retbins=True
    )
    bins_head, edges_head = pd.qcut(
        type_head_mean,
        q=q_head,
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
	# ContextID	Compound	Mod	Head	ModAvg	ModStd	HeadAvg	HeadStd	      Strata
    # 3470	13471	zinc plate	zinc	plate	3.1	2.469820	1.1	1.44914	    1_0
    # 3471	13472	zinc plate	zinc	plate	4.4	0.699206	2.5	1.64992	    1_0
    # 3472	13473	zinc plate	zinc	plate	3.2	2.097620	2.5	2.46080	    1_0
    # 3473	13474	zinc plate	zinc	plate	3.6	2.065590	2.6	2.11870	    1_0
    # 3474	13475	zinc plate	zinc	plate	4.4	0.843274	2.9	2.33095	    1_0
    # 3475	13476	zinc plate	zinc	plate	4.3	1.337490	3.0	2.05480	    1_0
    # 3476	13477	zinc plate	zinc	plate	3.3	1.494430	3.3	1.88856	    1_0
    # 3477	13478	zinc plate	zinc	plate	4.1	1.728840	3.3	1.94651	    1_0
    # 3478	13479	zinc plate	zinc	plate	4.8	0.421637	3.3	2.11082	    1_0
    # 3479	13480	zinc plate	zinc	plate	3.8	2.097620	3.5	1.35401	    1_0

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