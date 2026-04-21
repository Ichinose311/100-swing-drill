import MeCab

# ファイル読み込み
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

tagger = MeCab.Tagger() #形態素解析機を作る
node = tagger.parseToNode(text) #分解した単語がnodeとして連結されている

while node:
    # node.featureは次のようになっている
    # 動詞,自立,*,*,五段・ラ行,基本形,来る,キ,キ
    # これをカンマで分解してリストにする
    features = node.feature.split(",")
    if features[0] == "動詞": #feature[0]->品詞
        print(node.surface, "->", features[6]) #surfaceはそのままの単語,features[6]は基本形
    node = node.next
