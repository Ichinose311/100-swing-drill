import pandas as pd
# タブをスペースに置換して先頭10行を表示
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df.head(n=10).to_csv(sep=" ", index=False, header=None))