from pathlib import Path

import MeCab


def main():
    # ファイル読み込み
    with open(Path(__file__).with_name("text.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    tagger = MeCab.Tagger() #形態素解析機を作る
    node = tagger.parseToNode(text) #分解した単語がnodeとして連結されている

    while node:
        # pyproject.tomlで指定するUniDic-liteの素性列を参照する。
        # 0: 品詞、7: 語彙素（基本形）。IPADicでは列構成が異なる。
        features = node.feature.split(",")
        if features[0] == "動詞": #feature[0]->品詞
            print(node.surface, "->", features[7]) # UniDic-liteでは7列目（0始まり）が語彙素。IPADicの列番号とは異なる。
        node = node.next


if __name__ == "__main__":
    main()
