import pandas as pd

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。
uk_texts = uk_text.split("\n")
result = list(filter(lambda x: "[Category:" in x, uk_texts))
result = [a.replace("[[Category:", "").replace("|*", "").replace("]]", "") for a in result]
print(result)