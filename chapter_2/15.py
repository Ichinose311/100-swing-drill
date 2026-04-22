import pandas as pd

N = 10
#データ読み込み
df = pd.read_csv("popular-names.txt", sep="\t", header=None)
#一ファイルあたりの行数を計算
# //は整数除算
# 明確な単語を選ぶ（p.10）
chunk_size = len(df) // N
#ループで分割し切り出す範囲を決める
for i in range(N):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != N - 1 else len(df) # 最後のファイルは残りの行をすべて含むようにする
    #ilocで行番号を指定して切り出す
    chunk = df.iloc[start:end]
    #切り出したデータをファイルに出力
    # to_csv()でタブ区切りの文字列を出力
    # index=Falseで行番号を出力しない、header=Falseで列名を出力しない
    chunk.to_csv(f"out_{i}.txt", sep="\t", header=False, index=False)
#出力結果はout_0.txt, out_1.txt, ..., out_9.txtの10ファイルに分割される