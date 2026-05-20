import pandas as pd

df = pd.read_json('jawiki-country.json.gz', lines=True) 
#JSON形式のファイルを読み込む。lines=Trueは、1行ごとに1つのJSONデータが入っている形式
uk_text = df.query('title=="イギリス"')["text"].values[0] 
#title列がイギリスの行を抽出し、text列の値を取り出す。配列形式で最初の要素を取り出す。
uk_texts = uk_text.split("\n")
#改行ごとに分割してリストにする
result = list(filter(lambda x: "[Category:" in x, uk_texts))
#リストから[Category:を含む行だけを抽出する。filter関数とラムダ式を使う。
#filter(条件, リスト)は、リストの中から条件に合うものだけを残す
#lambda x: "[Category:" in xは、xが[Category:を含むかどうかを判定する関数
# list() は、filterの結果をリストに変換する
print(result)
#出力結果
'''
['[[Category:イギリス|*]]', '[[Category:イギリス連邦加盟国]]', '[[Category:英連邦王国|*]]', '[[Category:G8加盟国]]', '[[Category:欧州連合加盟国|元]]', '[[Category:海洋国家]]', '[[Category:現存する君主国]]', '[[Category:島国]]', '[[Category:1801年に成立した国家・領域]]']
'''

