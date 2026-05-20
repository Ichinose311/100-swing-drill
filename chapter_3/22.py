import pandas as pd

df = pd.read_json('jawiki-country.json.gz', lines=True) 
#JSON形式のファイルを読み込む。lines=Trueは、1行ごとに1つのJSONデータが入っている形式
uk_text = df.query('title=="イギリス"')["text"].values[0] 
#title列がイギリスの行を抽出し、text列の値を取り出す。配列形式で最初の要素を取り出す。
uk_texts = uk_text.split("\n")
result = list(filter(lambda x: "[Category:" in x, uk_texts))
result = [a.replace("[[Category:", "").replace("|*", "").replace("]]", "") for a in result]
# 先頭の "[[Category:" を削除
# "|*" を削除
# 末尾の "]]" を削除
# result の中身を1つずつ a に入れて処理する
print(result)
#出力結果
'''
['イギリス', 'イギリス連邦加盟国', '英連邦王国', 'G8加盟国', '欧州連合加盟国|元', '海洋国家', '現存する君主国', '島国', '1801年に成立した国家・領域']
'''