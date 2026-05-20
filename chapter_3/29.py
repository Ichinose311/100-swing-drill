import pandas as pd
import re
import requests

# requests.Session() は、APIにアクセスするための接続を管理するもの
# 1回だけなら requests.get() でもよいが、Sessionを使うと複数回アクセスするときに便利
S = requests.Session()

# Wikipedia APIのURL
# 今回は英語版WikipediaのAPIにアクセスする
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
# 基礎情報の中から「国旗画像」という項目の値を取り出す
# 例: "Flag of the United Kingdom.svg"
# split("|")[0] は、もし "|" が含まれていた場合に、その前だけを取り出すための処理
flag_file = result["国旗画像"].split("|")[0]
# Wikipedia APIで画像ファイルを指定するときは、ファイル名の前に "File:" をつける
# 例: "File:Flag of the United Kingdom.svg"
title = "File:" + flag_file
# Wikipedia APIに送る条件を辞書で指定する
PARAMS = {
    "action": "query",  # 情報を問い合わせる
    "format": "json",   # 結果をJSON形式で受け取る
    "prop": "imageinfo",# 画像情報を取得する
    "titles": title,     # 調べたい画像ファイル名
    "iiprop": "url" # 画像情報の中でもURLを取得する
}
# Wikipedia APIにリクエストを送る
R = S.get(
    url=URL,
    params=PARAMS,
    headers={
        # User-Agentは「誰がアクセスしているか」を示す情報
        # Wikipedia APIを使うときは、User-Agentを設定することが推奨されている
        "User-Agent": "IchinoseBot/1.0 (student)" #Wikipedia APIを利用する際には、User-Agentを設定することが推奨されている。ここでは、IchinoseBotという名前のBotを作成し、バージョン1.0であることを示している。また、studentというタグも付けている。
    }
)
# APIから返ってきたJSON形式のデータを、Pythonの辞書のような形に変換する
DATA = R.json()
# APIの返答から画像URLを取り出して表示する
# DATA["query"]["pages"] の中には、画像ファイルのページ情報が入っている
# .values() を使って、中身のページ情報だけを取り出す
for v in DATA["query"]["pages"].values():
    # imageinfo はリストになっているため [0] で最初の要素を取り出す
    # その中の "url" が、実際の画像URL
    print(v["imageinfo"][0]["url"])
#出力結果
'''
https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg
'''
