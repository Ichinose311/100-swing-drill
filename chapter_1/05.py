text = "I am an NLPer"
def n_gram(n, text):
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]
print(n_gram(2, text)) #bi-gram
print(n_gram(3, text)) #tri-gram