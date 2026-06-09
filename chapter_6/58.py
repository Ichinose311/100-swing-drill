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

    # 国名を抽出する対象セクション
    target_sections = {
        "capital-common-countries",
        "capital-world"
    }

    current_section = None
    countries = set()

    # questions-words.txt から国名を抽出
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith(":"):
                current_section = line[2:]
                continue

            if current_section not in target_sections:
                continue

            words = line.split()

            if len(words) != 4:
                continue

            word1, word2, word3, word4 = words

            # capital系のセクションでは、
            # 2列目と4列目が国名
            countries.add(word2)
            countries.add(word4)

    # モデルに存在する国名だけを使う
    country_names = []
    country_vectors = []

    for country in sorted(countries):
        if country in model.key_to_index:
            country_names.append(country)
            country_vectors.append(model[country])

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

#出力結果
'''
抽出した国名数: 116
デンドログラムを保存しました: /home/ichinose/projects/100-swing-drill/chapter_6/country_ward_dendrogram.png
'''