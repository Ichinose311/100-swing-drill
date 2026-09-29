from names_data import DATA_PATH, load_names


def main():
    df = load_names()
    # 明確な単語を選ぶ（p.10）
    # 3列目の数値で降順にソート
    df_sorted = df.sort_values(2, ascending=False)
    print(df_sorted)


if __name__ == "__main__":
    main()
