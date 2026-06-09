from pathlib import Path
from gensim.models import KeyedVectors

def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir / "GoogleNews-vectors-negative300.bin.gz"
    questions_path = base_dir / "questions-words.txt"
    output_path = base_dir / "analogy_result.txt"

    # 検索対象を上位30万語に制限する
    # 厳密に全語彙から探したい場合は None にする
    restrict_vocab = None

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True
    )

    # 類似度計算を少し高速化するためにノルムを事前計算
    model.fill_norms()

    current_section = None
    results = []

    count = 0

    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith(":"):
                current_section = line[2:]
                print(f"現在のセクション: {current_section}")
                continue

            words = line.split()

            if len(words) != 4:
                continue

            word1, word2, word3, word4 = words

            # 単語がモデルに存在しない場合はスキップ
            if word1 not in model.key_to_index:
                continue
            if word2 not in model.key_to_index:
                continue
            if word3 not in model.key_to_index:
                continue
            if word4 not in model.key_to_index:
                continue

            # vec(word2) - vec(word1) + vec(word3)
            similar_words = model.most_similar(
                positive=[word2, word3],
                negative=[word1],
                topn=1,
                restrict_vocab=restrict_vocab
            )

            predicted_word, similarity = similar_words[0]

            results.append(
                (
                    current_section,
                    word1,
                    word2,
                    word3,
                    word4,
                    predicted_word,
                    similarity
                )
            )

            count += 1

            # 進捗表示
            if count % 100 == 0:
                print(f"{count} 件処理しました")

    # 結果をファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "section\tword1\tword2\tword3\tword4\t"
            "predicted_word\tsimilarity\n"
        )

        for section, word1, word2, word3, word4, predicted_word, similarity in results:
            f.write(
                f"{section}\t{word1}\t{word2}\t{word3}\t{word4}\t"
                f"{predicted_word}\t{similarity}\n"
            )

    # 画面にも一部表示
    for result in results[:10]:
        section, word1, word2, word3, word4, predicted_word, similarity = result
        print(
            f"{section}: {word1} {word2} {word3} {word4} -> "
            f"{predicted_word} {similarity}"
        )

    print(f"\n結果を保存しました: {output_path}")
    print(f"処理した事例数: {len(results)}")

if __name__ == "__main__":
    main()

#出力結果
'''
19500 件処理しました
capital-common-countries: Athens Greece Baghdad Iraq -> Iraqi 0.635187029838562
capital-common-countries: Athens Greece Bangkok Thailand -> Thailand 0.7137669920921326
capital-common-countries: Athens Greece Beijing China -> China 0.7235778570175171
capital-common-countries: Athens Greece Berlin Germany -> Germany 0.6734622716903687
capital-common-countries: Athens Greece Bern Switzerland -> Switzerland 0.4919748306274414
capital-common-countries: Athens Greece Cairo Egypt -> Egypt 0.7527809739112854
capital-common-countries: Athens Greece Canberra Australia -> Australia 0.583732545375824
capital-common-countries: Athens Greece Hanoi Vietnam -> Viet_Nam 0.6276341676712036
capital-common-countries: Athens Greece Havana Cuba -> Cuba 0.6460990905761719
capital-common-countries: Athens Greece Helsinki Finland -> Finland 0.68999844789505

結果を保存しました: /home/ichinose/projects/100-swing-drill/chapter_6/analogy_result.txt
処理した事例数: 19544
'''