from names_data import DATA_PATH, load_names


def main():
    df = load_names()
    # 明確な単語を選ぶ（p.10）
    # 全体(100%ランダムに並び替え
    df_sample = df.sample(frac=1)

    # 保存
    df_sample.to_csv(DATA_PATH.parent / "shuffled.txt", sep="\t", header=False, index=False)
    # 出力結果はshuffled.txtにランダムに並び替えられたデータが保存される


if __name__ == "__main__":
    main()
