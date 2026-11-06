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

    for fold in range(1):
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
    for seed, fold, train_idx, val_idx in iter_folds(df, repeat=0):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # 1. In ra số lượng câu và compound trong tập train/valid
        # print(
        #     f"seed={seed}, fold={fold}, "
        #     f"train={len(train_df)} câu / {train_df['Compound'].nunique()} compound, "
        #     f"valid={len(val_df)} câu / {val_df['Compound'].nunique()} compound"
        # )

        # 2. In ra mean_ModAvg và mean_HeadAvg của tập train/valid
        # print(
        #     f"seed={seed}, fold={fold}, "
        #     f"mean_ModAvg={train_df['ModAvg'].mean():.2f}, "
        #     f"mean_HeadAvg={val_df['HeadAvg'].mean():.2f}"
        # )

        # 3. In ra các cặp (Mod, Head) trong tập valid mà không có compound nào trong tập train có cùng Mod hoặc Head
        # train_compounds = train_df[["Compound", "Mod", "Head"]].drop_duplicates()

        # rows = []

        # for mod, head in val_df[["Mod", "Head"]].drop_duplicates().itertuples(index=False):
        #     matching = (
        #         train_compounds.loc[
        #             train_compounds["Mod"].eq(mod) | train_compounds["Head"].eq(head),
        #             "Compound"
        #         ]
        #         .drop_duplicates()
        #         .sort_values()
        #         .tolist()
        #     )

        #     if matching:
        #         rows.append({
        #             "valid_pair": f"({mod}, {head})",
        #             "matching_train_compounds": ", ".join(matching),
        #         })

        # print(pd.DataFrame(rows).to_string(index=False))

        # 4. In ra số lượng câu cho mỗi compound trong tập train/valid
        # sentence_counts_train = train_df.groupby("Compound").size()
        # sentence_counts_val = val_df.groupby("Compound").size()
        # print("Số câu/compound : số compound")
        # print(sentence_counts_train.value_counts().sort_index().to_string())
        # print(sentence_counts_val.value_counts().sort_index().to_string())

if __name__ == "__main__":
    main()