import pandas as pd
# 末尾10行を表示
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df.tail(n=10))