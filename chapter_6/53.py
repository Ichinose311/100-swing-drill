from pathlib import Path
from gensim.models import KeyedVectors

def main():
    file_path = Path(__file__).parent / "GoogleNews-vectors-negative300.bin.gz"

    # Google Newsの学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        file_path,
        binary=True
    )

    # Spain - Madrid + Athens に近い単語を求める
    similar_words = model.most_similar(
        positive=["Spain", "Athens"],
        negative=["Madrid"],
        topn=10
    )

    print("Spain - Madrid + Athens と類似度が高い単語 上位10件:")
    for word, similarity in similar_words:
        print(f"{word}\t{similarity}")

if __name__ == "__main__":
    main()

#出力結果
