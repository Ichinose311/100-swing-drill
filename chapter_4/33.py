import CaboCha

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

parser = CaboCha.Parser()
tree = parser.parse(text) # 係り受け情報つきの構造に変える
# 分節を入れるリストと、現在の分節を入れる変数を用意する
chunks = []
current_chunk = None

# 文節を作る
for i in range(tree.size()): #単語ごとに見ていく
    token = tree.token(i)

    if token.chunk is not None: 
        # 新しい文節
        current_chunk = {
            "text": "",
            "link": token.chunk.link
        }
        chunks.append(current_chunk)

    if current_chunk is not None: #単語をつなげて文節を作る
        current_chunk["text"] += token.surface

# 係り受けを出力
for i, chunk in enumerate(chunks): #文節を1つずつ見る
    dst = chunk["link"] #係り先の文節番号
    if dst != -1 and dst < len(chunks): #文節じゃない、係り先が存在する
        print(chunk["text"], "\t", chunks[dst]["text"])

# 出力結果
'''
メロスは         激怒した。
激怒した。       決意した。
必ず、   除かなければならぬと
かの     邪智暴虐の
邪智暴虐の       王を
王を     除かなければならぬと
除かなければならぬと     決意した。
決意した。       わからぬ。
メロスには       わからぬ。
政治が   わからぬ。
わからぬ。       牧人である。
メロスは、       牧人である。
村の     牧人である。
牧人である。     暮して来た。
笛を     吹き、
吹き、   暮して来た。
羊と     遊んで
遊んで   暮して来た。
暮して来た。     敏感であった。
けれども         敏感であった。
邪悪に対しては、         敏感であった。
人一倍に         敏感であった。
'''