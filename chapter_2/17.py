import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 重複行の削除
print(df[0].unique())
