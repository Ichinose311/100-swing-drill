from wiki_article import load_article


def main():
    uk_text = load_article()
    uk_texts = uk_text.split("\n")
    result = list(filter(lambda x: "[Category:" in x, uk_texts))
    result = [a.replace("[[Category:", "").replace("|*", "").replace("]]", "") for a in result]
    # 先頭の "[[Category:" を削除
    # "|*" を削除
    # 末尾の "]]" を削除
    # result の中身を1つずつ a に入れて処理する
    print(result)


if __name__ == "__main__":
    main()
