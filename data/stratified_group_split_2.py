from sklearn.model_selection import StratifiedKFold
import pandas as pd

def split_data(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    comp_target = df.groupby('Compound')['ModAvg'].mean().reset_index()
    comp_target['mod_bin'] = pd.qcut(comp_target['ModAvg'], q=5, labels=False, duplicates='drop')
    
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(splitter.split(comp_target, comp_target["mod_bin"])):
        val_compounds = comp_target.iloc[val_idx]['Compound'].values
        df.loc[df['Compound'].isin(val_compounds), 'fold'] = fold
        
    return df
def main():
    path_dataset = "../dataset/en-nn-train.tsv"
    df = pd.read_csv(path_dataset, sep="\t", quoting=3, encoding="utf-8")

    df = split_data(df, n_splits=5, seed=42)

    print(df.groupby("fold")["Compound"].nunique())

    df.to_csv("../dataset/split_data.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
