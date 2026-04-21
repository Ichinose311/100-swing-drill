import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 1列目の出現回数をカウント
print(df[0].value_counts())
# 出力結果
#0
#James      118
#William    111
#John       108
#Robert     108
#Mary        92
#          ... 
#Scott        1
#Kelly        1
#Crystal      1
#Rachel       1
#Lucas        1
#Name: count, Length: 136, dtype: int64