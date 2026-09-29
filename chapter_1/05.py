from ngrams import generate_n_gram


def main():
    # 汎用コードをたくさん作る（p.135）

    text = "I am an NLPer"

    print(generate_n_gram(3, text))          # 文字tri-gram（空白も文字として保持）
    print(generate_n_gram(2, text.split()))  # 単語bi-gram


if __name__ == "__main__":
    main()
