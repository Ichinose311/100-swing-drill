import pandas as pd
# 末尾10行を表示
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df.tail(n=10))
# 出力結果
#             0  1      2     3
#2770      Liam  M  19837  2018
#2771      Noah  M  18267  2018
#2772   William  M  14516  2018
#2773     James  M  13525  2018
#2774    Oliver  M  13389  2018
#2775  Benjamin  M  13381  2018
#2776    Elijah  M  12886  2018
#2777     Lucas  M  12585  2018
#2778     Mason  M  12435  2018
#2779     Logan  M  12352  2018
