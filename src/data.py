import pandas as pd

path_dataset = "./data/raw/en-nn-train.tsv"
df = pd.read_csv(path_dataset, sep="\t", quoting=3, encoding="utf-8")

assert len(df) == 3480 and df["Compound"].nunique() == 444