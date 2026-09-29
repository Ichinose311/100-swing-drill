from country_vectors import load_country_vectors
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE


def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir / "GoogleNews-vectors-negative300.bin.gz"
    questions_path = base_dir / "questions-words.txt"
    output_path = base_dir / "country_tsne.png"
    result_path = base_dir / "country_tsne_result.txt"

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True
    )

    country_names, country_vectors = load_country_vectors(questions_path, model)

    country_vectors = np.array(country_vectors)

    print(f"抽出した国名数: {len(country_names)}")

    # t-SNEのperplexityはデータ数より小さくする必要がある
    perplexity = min(30, len(country_names) - 1)

    # t-SNEで2次元に圧縮
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=0,
        init="random",
        learning_rate="auto",
        max_iter=1000
    )

    country_vectors_2d = tsne.fit_transform(country_vectors)

    # 座標をファイルに保存
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("country\tx\ty\n")

        for country, vector_2d in zip(country_names, country_vectors_2d):
            x = vector_2d[0]
            y = vector_2d[1]
            f.write(f"{country}\t{x}\t{y}\n")

    # 可視化
    plt.figure(figsize=(16, 12))

    x_values = country_vectors_2d[:, 0]
    y_values = country_vectors_2d[:, 1]

    plt.scatter(x_values, y_values)

    for country, x, y in zip(country_names, x_values, y_values):
        plt.text(x, y, country, fontsize=8)

    plt.title("t-SNE Visualization of Country Word Vectors")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()

    # 画像として保存
    plt.savefig(output_path, dpi=300)

    print(f"t-SNEの座標を保存しました: {result_path}")
    print(f"t-SNEの可視化画像を保存しました: {output_path}")


if __name__ == "__main__":
    main()
