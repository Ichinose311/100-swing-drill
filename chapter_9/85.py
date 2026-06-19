from pathlib import Path
import csv
import zipfile

from transformers import AutoTokenizer


def find_file_in_zip(zip_file, target_name):
    """
    zip内から train.tsv / dev.tsv を探す
    """
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name

    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def load_sst2_tsv_from_zip(zip_path, target_name, tokenizer):
    """
    SST-2.zip から train.tsv / dev.tsv を読み込み、
    text, label, tokens を持つ辞書のリストに変換する
    """
    examples = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_path = find_file_in_zip(zf, target_name)

        with zf.open(tsv_path, "r") as f:
            # zip内のファイルは bytes として読まれるので、文字列に変換する
            lines = (line.decode("utf-8") for line in f)

            reader = csv.DictReader(lines, delimiter="\t")

            for row in reader:
                text = row["sentence"]
                label = int(row["label"])

                tokens = tokenizer.tokenize(text)

                example = {
                    "text": text,
                    "label": label,
                    "tokens": tokens,
                }

                examples.append(example)

    return examples


def show_examples(name, examples, n=3):
    """
    読み込んだデータの一部を表示する
    """
    print(f"===== {name} =====")
    print(f"事例数: {len(examples)}")
    print()

    for i, example in enumerate(examples[:n], start=1):
        print(f"--- 例 {i} ---")
        print("text  :", example["text"])
        print("label :", example["label"])
        print("tokens:", example["tokens"])
        print()


def main():
    base_dir = Path(__file__).parent

    sst_zip_path = base_dir / "SST-2.zip"

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    train_examples = load_sst2_tsv_from_zip(
        sst_zip_path,
        "train.tsv",
        tokenizer
    )

    dev_examples = load_sst2_tsv_from_zip(
        sst_zip_path,
        "dev.tsv",
        tokenizer
    )

    show_examples("train", train_examples)
    show_examples("dev", dev_examples)


if __name__ == "__main__":
    main()

#出力結果
'''
===== train =====
事例数: 67349

--- 例 1 ---
text  : hide new secretions from the parental units 
label : 0
tokens: ['hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units']

--- 例 2 ---
text  : contains no wit , only labored gags 
label : 0
tokens: ['contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s']

--- 例 3 ---
text  : that loves its characters and communicates something rather beautiful about human nature 
label : 1
tokens: ['that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature']

===== dev =====
事例数: 872

--- 例 1 ---
text  : it 's a charming and often affecting journey. 
label : 1
tokens: ['it', "'", 's', 'a', 'charming', 'and', 'often', 'affecting', 'journey', '.']

--- 例 2 ---
text  : unflinchingly bleak and desperate 
label : 0
tokens: ['un', '##fl', '##in', '##ching', '##ly', 'bleak', 'and', 'desperate']

--- 例 3 ---
text  : allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . 
label : 1
tokens: ['allows', 'us', 'to', 'hope', 'that', 'nolan', 'is', 'poised', 'to', 'embark', 'a', 'major', 'career', 'as', 'a', 'commercial', 'yet', 'in', '##vent', '##ive', 'filmmaker', '.']
'''