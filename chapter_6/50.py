from pathlib import Path
from gensim.models import KeyedVectors

def main():
    # Pythonファイルと同じディレクトリにあるデータセットを指定
    file_path = Path(__file__).parent / "GoogleNews-vectors-negative300.bin.gz"

    # Google Newsのword2vec形式のバイナリを読み込む
    model = KeyedVectors.load_word2vec_format(
        file_path,
        binary=True
    )

    # "United States" は内部では "United_States" と表現されている
    word = "United_States"

    # 単語ベクトルを取得
    vector = model[word]

    print(f"{word} の単語ベクトル:")
    print(vector)

    print("\nベクトルの次元数:")
    print(vector.shape)

if __name__ == "__main__":
    main()

#出力結果
