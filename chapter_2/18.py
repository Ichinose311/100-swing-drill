from names_data import DATA_PATH, load_names


def main():
    df = load_names()
    # 明確な単語を選ぶ（p.10）
    # 1列目の出現回数をカウント
    value_counts = df[0].value_counts()
    print(value_counts)


if __name__ == "__main__":
    main()
