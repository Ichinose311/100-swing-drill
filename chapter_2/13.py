import pandas as pd
# タブをスペースに置換して先頭10行を表示
#sep='\t'  # タブ区切り
#header=None  # ヘッダーなし
# 明確な単語を選ぶ（p.10）
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
# df.head(n=10) # 先頭10行を表示
# .to csv()でスペース区切りの文字列を出力(通常はカンマ,)
# index=Falseで行番号を出力しない、header=Noneで列名を出力しない
print(df.head(n=10).to_csv(sep=" ", index=False, header=None))
# 出力結果
#Mary F 7065 1880
#Anna F 2604 1880
#Emma F 2003 1880
#Elizabeth F 1939 1880
#Minnie F 1746 1880
#Margaret F 1578 1880
#Ida F 1472 1880
#Alice F 1414 1880
#Bertha F 1320 1880
#Sarah F 1288 1880