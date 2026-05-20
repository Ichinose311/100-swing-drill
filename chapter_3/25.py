import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。
uk_texts = uk_text.split("\n")
# 基礎情報テンプレートの「|項目名 = 値」の行を探すための正規表現を作る
# \| : ｜からはじまる行を対象にする
# (.+?) : 1文字以上の任意の文字を非貪欲マッチで取得する
# \s=\s* : =の前後に0個以上の空白があることを許容する
# (.+) : 1文字以上の任意の文字を貪欲マッチで取得する
# |国名 = イギリス みたいな行からキー:国名、値:イギリスを抽出するための正規表現
pattern = re.compile("\|(.+?)\s=\s*(.+)")
# 取り出した項目名と値を入れるための空の辞書を作る
result = {}
for line in uk_texts:
    # 現在の1行が、「|項目名 = 値」の形になっているか調べる
    r = re.search(pattern, line)
    if r:
        # r[1] には項目名が入っている
        # r[2] には値が入っている
        # 例: 「|首都 = ロンドン」なら result["首都"] = "ロンドン"
        result[r[1]] = r[2]

print(result)
#出力結果
