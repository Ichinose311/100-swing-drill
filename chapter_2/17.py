import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 重複行の削除
print(df[0].unique())
# 出力結果
#[     'Mary',      'Anna',      'Emma', 'Elizabeth',    'Minnie',  'Margaret',
#       'Ida',     'Alice',    'Bertha',     'Sarah',
# ...
#     'Mason',      'Liam', 'Charlotte',    'Harper',  'Benjamin',    'Elijah',
#    'Amelia',     'Logan',    'Oliver',     'Lucas']
#Length: 136, dtype: str