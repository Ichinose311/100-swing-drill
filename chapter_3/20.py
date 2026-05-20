import pandas as pd

df = pd.read_json('jawiki-country.json.gz', lines=True) 
#JSON形式のファイルを読み込む。lines=Trueは、1行ごとに1つのJSONデータが入っている形式
uk_text = df.query('title=="イギリス"')["text"].values[0] 
#title列がイギリスの行を抽出し、text列の値を取り出す。配列形式で最初の要素を取り出す。
print(uk_text)
#出力結果
'''
省略
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
'''