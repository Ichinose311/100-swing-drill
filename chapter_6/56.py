from pathlib import Path
import csv
import zipfile
import io

from gensim.models import KeyedVectors
from scipy.stats import spearmanr


def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir / "GoogleNews-vectors-negative300.bin.gz"
    wordsim_path = base_dir / "wordsim353.zip"
    output_path = base_dir / "wordsim353_result.txt"

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True
    )

    # wordsim353.zip 内の combined.csv を探す
    with zipfile.ZipFile(wordsim_path, "r") as z:
        csv_filename = None

        for name in z.namelist():
            if name.endswith("combined.csv"):
                csv_filename = name
                break

        if csv_filename is None:
            raise FileNotFoundError("wordsim353.zip の中に combined.csv が見つかりません")

        results = []
        human_scores = []
        vector_scores = []

        with z.open(csv_filename) as f:
            text_file = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.DictReader(text_file)

            for row in reader:
                word1 = row["Word 1"]
                word2 = row["Word 2"]
                human_score = float(row["Human (mean)"])

                # GoogleNewsでは複合語が _ で表される場合がある
                word1_key = word1.replace(" ", "_")
                word2_key = word2.replace(" ", "_")

                # 単語がモデルに存在しない場合はスキップ
                if word1_key not in model.key_to_index:
                    continue
                if word2_key not in model.key_to_index:
                    continue

                # 単語ベクトルによるコサイン類似度
                vector_similarity = model.similarity(word1_key, word2_key)

                human_scores.append(human_score)
                vector_scores.append(vector_similarity)

                results.append(
                    (
                        word1,
                        word2,
                        human_score,
                        vector_similarity
                    )
                )

    # スピアマン相関係数を計算
    correlation, p_value = spearmanr(human_scores, vector_scores)

    # 結果をファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("word1\tword2\thuman_score\tvector_similarity\n")

        for word1, word2, human_score, vector_similarity in results:
            f.write(
                f"{word1}\t{word2}\t{human_score}\t{vector_similarity}\n"
            )

    print("WordSimilarity-353での評価結果")
    print(f"使用した単語ペア数: {len(results)}")
    print(f"スピアマン相関係数: {correlation}")
    print(f"p値: {p_value}")
    print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()

#出力結果
'''
WordSimilarity-353での評価結果
使用した単語ペア数: 353
スピアマン相関係数: 0.7000166486272194
p値: 2.86866666051422e-53

結果を保存しました: /home/ichinose/projects/100-swing-drill/chapter_6/wordsim353_result.txt
'''