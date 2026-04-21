import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 3列目の数値で降順にソート
print(df.sort_values(2, ascending=False))
# 出力結果
#            0  1      2     3
#1340    Linda  F  99689  1947
#1360    Linda  F  96211  1948
#1350    James  M  94757  1947
#1550  Michael  M  92704  1957
#1351   Robert  M  91640  1947
#...       ... ..    ...   ...
#27      Annie  F   1326  1881
#28     Bertha  F   1324  1881
#8      Bertha  F   1320  1880
#29      Alice  F   1308  1881
#9       Sarah  F   1288  1880

#[2780 rows x 4 columns]
