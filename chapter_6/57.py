from country_vectors import load_country_vectors
from pathlib import Path

from gensim.models import KeyedVectors
from sklearn.cluster import KMeans


def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir / "GoogleNews-vectors-negative300.bin.gz"
    questions_path = base_dir / "questions-words.txt"
    output_path = base_dir / "country_kmeans_result.txt"

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True
    )

    country_names, country_vectors = load_country_vectors(questions_path, model)

    print(f"抽出した国名数: {len(country_names)}")

    # k-meansクラスタリング
    kmeans = KMeans(
        n_clusters=5,
        random_state=0,
        n_init=10
    )

    labels = kmeans.fit_predict(country_vectors)

    # 結果をファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("country\tcluster\n")

        for country, label in zip(country_names, labels):
            f.write(f"{country}\t{label}\n")

    # クラスタごとに表示
    for cluster_id in range(5):
        print()
        print(f"クラスタ {cluster_id}")

        for country, label in zip(country_names, labels):
            if label == cluster_id:
                print(country)

    print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
