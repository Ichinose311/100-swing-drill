from pathlib import Path
from gensim.models import KeyedVectors

def main():
    file_path = Path(__file__).parent / "GoogleNews-vectors-negative300.bin.gz"

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        file_path,
        binary=True
    )

    word1 = "United_States"
    word2 = "U.S."

    # コサイン類似度を計算
    similarity = model.similarity(word1, word2)

    print(f"{word1} と {word2} のコサイン類似度:")
    print(similarity)

if __name__ == "__main__":
    main()

#出力結果
