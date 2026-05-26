import CaboCha

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

parser = CaboCha.Parser()
tree = parser.parse(text)

chunks = []
current_chunk = None

# 文節作成
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

# 主語「メロス」を含む文節を探す
for i, chunk in enumerate(chunks):
    if "メロス" in chunk["text"]:
        dst = chunk["link"]

        if dst != -1 and dst < len(chunks):
            print("主語:", chunk["text"])
            print("述語:", chunks[dst]["text"])
            print()

# 出力結果
'''
主語: メロスは
述語: 激怒した。

主語: メロスには
述語: わからぬ。

主語: メロスは、
述語: 牧人である。
'''