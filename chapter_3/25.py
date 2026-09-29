from wiki_article import load_article

import re


def main():
    uk_text = load_article()
    uk_texts = uk_text.split("\n")
    # 基礎情報テンプレートの「|項目名 = 値」の行を探すための正規表現を作る
    # \| : ｜からはじまる行を対象にする
    # (.+?) : 1文字以上の任意の文字を非貪欲マッチで取得する
    # \s=\s* : =の前後に0個以上の空白があることを許容する
    # (.+) : 1文字以上の任意の文字を貪欲マッチで取得する
    # |国名 = イギリス みたいな行からキー:国名、値:イギリスを抽出するための正規表現
    pattern = re.compile(r"\|(.+?)\s=\s*(.+)")
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


if __name__ == "__main__":
    main()
