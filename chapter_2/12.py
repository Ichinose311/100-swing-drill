from names_data import DATA_PATH, load_names


def main():
    # 末尾10行を表示
    #sep='\t'  # タブ区切り
    #header=None  # ヘッダーなし
    # 明確な単語を選ぶ（p.10）
    df = load_names()
    print(df.tail(n=10))


if __name__ == "__main__":
    main()
