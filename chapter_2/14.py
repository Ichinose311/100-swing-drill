import pandas as pd
# 先頭10行1列目を表示
#sep='\t'  # タブ区切り
#header=None  # ヘッダーなし
# 明確な単語を選ぶ（p.10）
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df[0].head(n=10))
# 出力結果
# 0         Mary
# 1         Anna
# 2         Emma
# 3    Elizabeth
# 4      Minnie
# 5    Margaret
# 6         Ida
# 7       Alice
# 8       Bertha
# 9       Sarah
# Name: 0, dtype: str