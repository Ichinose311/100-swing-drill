# 汎用コードをたくさん作る（p.135）
def generate_n_gram(n, text):
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]

text1 = "paraparaparadise"
text2 = "paragraph"

# 名前のフォーマットで情報を伝える（p.25）
set_x = set(generate_n_gram(2, text1))
set_y = set(generate_n_gram(2, text2))

print(set_x | set_y)  # 和集合
print(set_x & set_y)  # 積集合
print(set_x - set_y)  # 差集合
print('se' in set_x or 'se' in set_y)
