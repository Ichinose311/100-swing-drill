from wiki_article import load_article

import re


def main():
    uk_text = load_article()

    for section in re.findall(r"(=+)([^=]+)\1\n", uk_text):
        #re.findall(pattern, string)は、stringの中からpatternにマッチする部分をすべて抽出してリストで返す。
        #patternは、=で囲まれた文字列を抽出する正規表現。
        #(=+)は、=が1回以上続く部分にマッチする正規表現。これをグループ1とする。
        #([^=]+)は、=以外の文字が1回以上続く部分にマッチする正規表現。これをグループ2とする。
        #\1は、グループ1と同じ文字列にマッチする正規表現。つまり、=で囲まれた文字列を抽出するための正規表現。
        #\nは、改行にマッチする正規表現。
        #sectionは、(グループ1, グループ2)のタプルである。グループ1は=の数、グループ2はセクション名である。
        print(f"{section[1].strip()}\t{len(section[0]) - 1}")
        #section[1].strip()は、セクション名の前後の空白を削除する。len(section[0]) - 1は、=の数から1を引いた値である。これがセクションの階層を表す。
        #出力結果は、セクション名と階層をタブ区切りで表示する。
        #=の数が2ならレベル1



if __name__ == "__main__":
    main()
