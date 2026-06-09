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
'''
Spain - Madrid + Athens と類似度が高い単語 上位10件:
Greece  0.6898480653762817
Aristeidis_Grigoriadis  0.560684859752655
Ioannis_Drymonakos      0.5552908778190613
Greeks  0.545068621635437
Ioannis_Christou        0.5400862097740173
Hrysopiyi_Devetzi       0.5248444676399231
Heraklio        0.5207759737968445
Athens_Greece   0.516880989074707
Lithuania       0.5166865587234497
Iraklion        0.5146791338920593
'''