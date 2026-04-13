import pandas as pd
# 先頭10行を表示
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df.head(n=10))
