from names_data import DATA_PATH, load_names


def main():
    df = load_names()
    # 明確な単語を選ぶ（p.10）
    # 重複行の削除
    unique_names = df[0].unique()
    print(unique_names)


if __name__ == "__main__":
    main()
