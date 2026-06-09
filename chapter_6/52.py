from pathlib import Path
from gensim.models import KeyedVectors

def main():
    file_path = Path(__file__).parent / "GoogleNews-vectors-negative300.bin.gz"

    # Google Newsの学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        file_path,
        binary=True
    )

    word = "United_States"

    # 類似度が高い単語を上位10件取得
    similar_words = model.most_similar(word, topn=10)

    print(f"{word} とコサイン類似度が高い単語 上位10件:")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}\t{similarity}")

if __name__ == "__main__":
    main()

#出力結果
