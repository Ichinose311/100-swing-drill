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
'''
United_States とコサイン類似度が高い単語 上位10件:
Unites_States   0.7877248525619507
Untied_States   0.7541370987892151
United_Sates    0.7400724291801453
U.S.    0.7310774326324463
theUnited_States        0.6404393911361694
America 0.6178410053253174
UnitedStates    0.6167312264442444
Europe  0.6132988929748535
countries       0.6044804453849792
Canada  0.601906955242157
'''