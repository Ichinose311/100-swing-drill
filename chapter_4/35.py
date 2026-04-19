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
