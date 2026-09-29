from wiki_article import load_article

import re

def remove_emphasis(dc):
    r = re.compile("'+")
    # 'が1回以上続く部分にマッチする正規表現
    return {k: r.sub("", v) for k, v in dc.items()}


def main():
    # 辞書の各値に対して、正規表現rを使って、'を空文字に置換する。辞書内包表記を使って、新しい辞書を作成する。

    uk_text = load_article()
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
    print(remove_emphasis(result))


if __name__ == "__main__":
    main()
