import MeCab

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

tagger = MeCab.Tagger()
node = tagger.parseToNode(text)

# 形態素を順番に保存(surfaceと品詞用のリスト)
words = []

while node:
    features = node.feature.split(",")
    words.append({
        "surface": node.surface,
        "pos": features[0]
    })
    node = node.next

# 「名詞 の 名詞」を探す
for i in range(len(words) - 2):
    if (
        words[i]["pos"] == "名詞" and
        words[i + 1]["surface"] == "の" and
        words[i + 1]["pos"] == "助詞" and
        words[i + 2]["pos"] == "名詞"
    ):
        phrase = words[i]["surface"] + words[i + 1]["surface"] + words[i + 2]["surface"]
        print(phrase)