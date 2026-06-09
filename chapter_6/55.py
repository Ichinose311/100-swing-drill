from pathlib import Path
import csv

def main():
    base_dir = Path(__file__).parent

    result_path = base_dir / "analogy_result.txt"

    semantic_total = 0
    semantic_correct = 0

    syntactic_total = 0
    syntactic_correct = 0

    with open(result_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            section = row["section"]
            word4 = row["word4"]
            predicted_word = row["predicted_word"]

            if word4 == predicted_word:
                is_correct = True
            else:
                is_correct = False

            # gramで始まるセクションは文法的アナロジー
            if section.startswith("gram"):
                syntactic_total += 1

                if is_correct:
                    syntactic_correct += 1

            # それ以外は意味的アナロジー
            else:
                semantic_total += 1

                if is_correct:
                    semantic_correct += 1

    semantic_accuracy = semantic_correct / semantic_total
    syntactic_accuracy = syntactic_correct / syntactic_total

    print("意味的アナロジー semantic analogy")
    print(f"正解数: {semantic_correct}")
    print(f"全問題数: {semantic_total}")
    print(f"正解率: {semantic_accuracy}")

    print()

    print("文法的アナロジー syntactic analogy")
    print(f"正解数: {syntactic_correct}")
    print(f"全問題数: {syntactic_total}")
    print(f"正解率: {syntactic_accuracy}")

if __name__ == "__main__":
    main()

#出力結果
'''
意味的アナロジー semantic analogy
正解数: 6482
全問題数: 8869
正解率: 0.7308602999210734

文法的アナロジー syntactic analogy
正解数: 7900
全問題数: 10675
正解率: 0.7400468384074942
'''