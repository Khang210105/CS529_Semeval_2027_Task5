from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def main():
    path_train = Path("../dataset/en-nn-train.tsv")
    df_train = pd.read_csv(path_train, sep="\t", header=0, quoting=3, encoding="utf-8")

    path_trial = Path("../dataset/en-nn-trial.tsv")
    df_trial = pd.read_csv(path_trial, sep="\t", header=0, quoting=3, encoding="utf-8")

    df = pd.concat([df_train, df_trial], ignore_index=True)

    freq = df['Compound'].value_counts()

    train_compounds = freq[freq >= 6].index
    valid_compounds = freq[freq < 6].index

    df_train = df[df['Compound'].isin(train_compounds)]
    df_valid = df[df['Compound'].isin(valid_compounds)]

    df_train.to_csv('../dataset/train/train.tsv', sep='\t', index=False)
    df_valid.to_csv('../dataset/valid/valid.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()