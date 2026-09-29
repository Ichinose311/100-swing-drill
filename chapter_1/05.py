def generate_n_gram(n, text):
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]


def main():
    # 汎用コードをたくさん作る（p.135）

    text = "I am an NLPer"

    print(generate_n_gram(2, text))  # bi-gram
    print(generate_n_gram(3, text))  # tri-gram


if __name__ == "__main__":
    main()
