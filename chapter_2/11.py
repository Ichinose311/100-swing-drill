import pandas as pd
# 先頭10行を表示
df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(df.head(n=10))
# 出力結果
#            0  1     2     3
# 0       Mary  F  7065  1880
# 1       Anna  F  2604  1880
# 2       Emma  F  2003  1880
# 3  Elizabeth  F  1939  1880
# 4     Minnie  F  1746  1880
# 5   Margaret  F  1578  1880
# 6        Ida  F  1472  1880
# 7      Alice  F  1414  1880
