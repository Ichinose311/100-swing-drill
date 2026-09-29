from country_vectors import load_country_vectors
from pathlib import Path

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram


def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir / "GoogleNews-vectors-negative300.bin.gz"
    questions_path = base_dir / "questions-words.txt"
    output_path = base_dir / "country_ward_dendrogram.png"

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True
    )

    country_names, country_vectors = load_country_vectors(questions_path, model)

    print(f"抽出した国名数: {len(country_names)}")

    # Ward法による階層型クラスタリング
    linkage_result = linkage(
        country_vectors,
        method="ward"
    )

    # デンドログラムを描画
    plt.figure(figsize=(20, 8))

    dendrogram(
        linkage_result,
        labels=country_names,
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title("Ward Hierarchical Clustering of Country Word Vectors")
    plt.xlabel("Country")
    plt.ylabel("Distance")
    plt.tight_layout()

    # 画像として保存
    plt.savefig(output_path, dpi=300)

    print(f"デンドログラムを保存しました: {output_path}")


if __name__ == "__main__":
    main()
