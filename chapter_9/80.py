from transformers import AutoTokenizer


def main():
    text = "The movie was full of incomprehensibilities."

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    tokens = tokenizer.tokenize(text)

    print(tokens)


if __name__ == "__main__":
    main()

#出力結果
'''
['the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.']
'''