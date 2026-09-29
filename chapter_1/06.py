from ngrams import generate_n_gram


def main():
    # 汎用コードをたくさん作る（p.135）

    text1 = "paraparaparadise"
    text2 = "paragraph"

    # 名前のフォーマットで情報を伝える（p.25）
    set_x = set(generate_n_gram(2, text1))
    set_y = set(generate_n_gram(2, text2))

    print(set_x | set_y)  # 和集合
    print(set_x & set_y)  # 積集合
    print(set_x - set_y)  # 差集合
    print("X contains se:", 'se' in set_x)
    print("Y contains se:", 'se' in set_y)


if __name__ == "__main__":
    main()
