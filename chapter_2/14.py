from names_data import DATA_PATH, load_names


def main():
    # 先頭10行1列目を表示
    #sep='\t'  # タブ区切り
    #header=None  # ヘッダーなし
    # 明確な単語を選ぶ（p.10）
    df = load_names()
    print(df[0].head(n=10))


if __name__ == "__main__":
    main()
