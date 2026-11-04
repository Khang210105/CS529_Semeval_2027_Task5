from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np

RANDOM_SEEDS = [23520705, 22521320, 23521413]
N_SPLITS = 5

def build_strata(df, n_bins=5):
    type_mod_mean = df.groupby("Compound")["ModAvg"].mean()

    q = min(n_bins, type_mod_mean.nunique())

    bins = pd.qcut(
        type_mod_mean,
        q=q,
        labels=False,
        duplicates="drop"
    )

    return df["Compound"].map(bins).astype(int)


def make_group_folds(data, seeds=RANDOM_SEEDS, n_splits=N_SPLITS):
    data = data.reset_index(drop=True).copy()
    strata = build_strata(data)

    all_folds = {}
    manifests = []

    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        seed_folds = []

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(data, y=strata, groups=data["Compound"])
        ):
            train_groups = set(data.iloc[train_idx]["Compound"])
            val_groups = set(data.iloc[val_idx]["Compound"])

            assert train_groups.isdisjoint(val_groups)

            seed_folds.append((train_idx, val_idx))

            fold_manifest = data[["ContextID", "Compound"]].copy()
            fold_manifest["seed"] = seed
            fold_manifest["fold"] = fold
            fold_manifest["split"] = "train"

            fold_manifest.loc[val_idx, "split"] = "validation"
            manifests.append(fold_manifest)

        all_folds[seed] = seed_folds

    manifest = pd.concat(manifests, ignore_index=True)

    return all_folds, manifest

path_dataset = "./dataset/en-nn-train.tsv"
df = pd.read_csv(path_dataset, sep="\t", quoting=3, encoding="utf-8")

def main():

    folds, manifest = make_group_folds(df)

    manifest.to_csv(
        "split_manifest.csv",
        index=False
    )

    ACTIVE_SEED = 23520705
    ACTIVE_FOLD = 0

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