import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 3列目の数値で降順にソート
print(df.sort_values(2, ascending=False))
