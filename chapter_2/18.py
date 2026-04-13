import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 1列目の出現回数をカウント
print(df[0].value_counts())