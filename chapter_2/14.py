import pandas as pd
# 先頭10行1列目を表示
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df[0].head(n=10))
