import pandas as pd
import re
import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。
uk_texts = uk_text.split("\n")
# \| : ｜からはじまる行を対象にする
# (.+?) : 1文字以上の任意の文字を非貪欲マッチで取得する
# \s=\s* : =の前後に0個以上の空白があることを許容する
# (.+) : 1文字以上の任意の文字を貪欲マッチで取得する
# |国名 = イギリス みたいな行からキー:国名、値:イギリスを抽出するための正規表現
pattern = re.compile(r"\|(.+?)\s=\s*(.+)")
result = {}
for line in uk_texts:
    r = re.search(pattern, line)
    if r:
        result[r[1]] = r[2]

flag_file = result["国旗画像"].split("|")[0]
title = "File:" + flag_file

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": title,
    "iiprop": "url"
}

R = S.get(
    url=URL,
    params=PARAMS,
    headers={
        "User-Agent": "IchinoseBot/1.0 (student)" #Wikipedia APIを利用する際には、User-Agentを設定することが推奨されている。ここでは、IchinoseBotという名前のBotを作成し、バージョン1.0であることを示している。また、studentというタグも付けている。
    }
)

DATA = R.json()

for v in DATA["query"]["pages"].values():
    print(v["imageinfo"][0]["url"])
