# 汎用コードをたくさん作る（p.135）
def generate_n_gram(n, text):
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]

text = "I am an NLPer"

print(generate_n_gram(2, text))  # bi-gram
print(generate_n_gram(3, text))  # tri-gram
