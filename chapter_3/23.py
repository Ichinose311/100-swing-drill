import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。

for section in re.findall(r"(=+)([^=]+)\1\n", uk_text):
    print(f"{section[1].strip()}\t{len(section[0]) - 1}")