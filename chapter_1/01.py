


def main():
    # 定数にコメントをつける（p.62）
    text = 'パタトクカシーー'
    step = 2  # 2文字おきに抽出する

    print(text[1::step])  # 2, 4, 6, 8文字目（0始まりでは1, 3, 5, 7）


if __name__ == "__main__":
    main()
