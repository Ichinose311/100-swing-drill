import pandas as pd

df = pd.read_csv("popular-names.txt", sep="\t", header=None)
# 明確な単語を選ぶ（p.10）
# 重複行の削除
unique_names = df[0].unique()
print(unique_names)
# 出力結果
#[     'Mary',      'Anna',      'Emma', 'Elizabeth',    'Minnie',  'Margaret',
#       'Ida',     'Alice',    'Bertha',     'Sarah',
# ...
#     'Mason',      'Liam', 'Charlotte',    'Harper',  'Benjamin',    'Elijah',
#    'Amelia',     'Logan',    'Oliver',     'Lucas']
#Length: 136, dtype: str