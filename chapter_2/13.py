from names_data import DATA_PATH, load_names


def main():
    # タブをスペースに置換して先頭10行を表示
    #sep='\t'  # タブ区切り
    #header=None  # ヘッダーなし
    # 明確な単語を選ぶ（p.10）
    df = load_names()
    # df.head(n=10) # 先頭10行を表示
    # .to csv()でスペース区切りの文字列を出力(通常はカンマ,)
    # index=Falseで行番号を出力しない、header=Noneで列名を出力しない
    print(df.head(n=10).to_csv(sep=" ", index=False, header=None))


if __name__ == "__main__":
    main()
