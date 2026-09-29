from wiki_article import load_article


def main():
    uk_text = load_article()
    uk_texts = uk_text.split("\n")
    #改行ごとに分割してリストにする
    result = list(filter(lambda x: "[Category:" in x, uk_texts))
    #リストから[Category:を含む行だけを抽出する。filter関数とラムダ式を使う。
    #filter(条件, リスト)は、リストの中から条件に合うものだけを残す
    #lambda x: "[Category:" in xは、xが[Category:を含むかどうかを判定する関数
    # list() は、filterの結果をリストに変換する
    print(result)


if __name__ == "__main__":
    main()
