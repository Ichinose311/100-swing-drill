import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)

# 全体(100%ランダムに並び替え
df = df.sample(frac=1)

# 保存
df.to_csv("shuffled.txt", sep="\t", header=False, index=False)
# 出力結果はshuffled.txtにランダムに並び替えられたデータが保存される