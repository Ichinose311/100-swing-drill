import pandas as pd

N = 10
#データ読み込み
df = pd.read_csv("popular-names.txt", sep="\t", header=None)
#一つあたりの行数を計算
chunk_size = len(df) // N
#ループで分割し切り出す範囲を決める
for i in range(N):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != N - 1 else len(df)

    chunk = df.iloc[start:end]
    chunk.to_csv(f"out_{i}.txt", sep="\t", header=False, index=False)