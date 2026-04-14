# 一つの処理を一行で詰め込みすぎない（p.64）
text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

clean_text = text.replace(",", "").replace(".", "")
words = clean_text.split()

# 意味のある名前を使う（p.18）
word_lengths = [len(word) for word in words]

print(word_lengths)
