import CaboCha

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

parser = CaboCha.Parser()
tree = parser.parse(text)

chunks = []
current_chunk = None

for i in range(tree.size()):
    token = tree.token(i)

    if token.chunk is not None:
        current_chunk = {
            "text": "",
            "link": token.chunk.link
        }
        chunks.append(current_chunk)

    if current_chunk is not None:
        current_chunk["text"] += token.surface

# 可視化
for i, chunk in enumerate(chunks):
    if chunk["link"] != -1:
        print(f"{chunk['text']} → {chunks[chunk['link']]['text']}")

# 出力結果
'''
メロスは → 激怒した。
激怒した。 → 決意した。
必ず、 → 除かなければならぬと
かの → 邪智暴虐の
邪智暴虐の → 王を
王を → 除かなければならぬと
除かなければならぬと → 決意した。
決意した。 → わからぬ。
メロスには → わからぬ。
政治が → わからぬ。
わからぬ。 → 牧人である。
メロスは、 → 牧人である。
村の → 牧人である。
牧人である。 → 暮して来た。
笛を → 吹き、
吹き、 → 暮して来た。
羊と → 遊んで
遊んで → 暮して来た。
暮して来た。 → 敏感であった。
けれども → 敏感であった。
邪悪に対しては、 → 敏感であった。
人一倍に → 敏感であった。
'''