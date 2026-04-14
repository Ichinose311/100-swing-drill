# 明確な単語を選ぶ（p.10）
text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

clean_text = text.replace(",", "").replace(".", "")
words = clean_text.split()

# 名前のフォーマットで情報を伝える（p.25）
word_lengths = [len(word) for word in words]

print(word_lengths)
