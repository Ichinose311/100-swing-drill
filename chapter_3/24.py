from wiki_article import load_article

import re


def main():
    uk_text = load_article()
    # re.fiandall(パターン,文字列) マッチした部分をリストで返す
    # 正規表現
    # \[\[ : [[Wikiのリンクの開始
    # (ファイル|File) : ファイルまたはFileのどちらかにマッチ
    # : : コロン
    # ([^]|]+?) : |または]以外の文字が1回以上
    # (\|.*?)+ : |に続く任意の文字が0回以上
    # \]\] : Wikiのリンクの終了
    for file in re.findall(r"\[\[(ファイル|File):([^]|]+?)(\|.*?)+\]\]", uk_text):
        # findallの戻り値は、マッチした部分のグループごとのタプルのリスト
        '''[
                ('ファイル', 'Example.jpg', '|thumb|200px'),
                ('File', 'Image.png', '|right')
            ]'''
        # file = ('ファイル', 'Example.jpg', '|thumb') インデックス1のとき
        # file[1] = 'Example.jpg'
        print(file[1])


if __name__ == "__main__":
    main()
