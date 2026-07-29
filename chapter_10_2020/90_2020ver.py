from pathlib import Path
import tarfile

from janome.tokenizer import Tokenizer as JanomeTokenizer
from nltk.tokenize import TreebankWordTokenizer


# ============================================================
# パスの設定
# ============================================================

# このPythonファイルが置かれているディレクトリ
BASE_DIR = Path(__file__).resolve().parent

# 同じディレクトリに置かれているKFTTの圧縮ファイル
ARCHIVE_PATH = BASE_DIR / "kftt-data-1.0.tar.gz"

# 展開後のデータセットディレクトリ
KFTT_DIR = BASE_DIR / "kftt-data-1.0"

# 元データが格納されているディレクトリ
ORIG_DIR = KFTT_DIR / "data" / "orig"

# 前処理後のデータを保存するディレクトリ
OUTPUT_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# トークナイザの準備
# ============================================================

# 日本語：形態素解析
ja_tokenizer = JanomeTokenizer()

# 英語：単語単位のトークン化
# TreebankWordTokenizerは追加データのダウンロードが不要
en_tokenizer = TreebankWordTokenizer()


def extract_dataset() -> None:
    """
    kftt-data-1.0.tar.gzを展開する。
    すでに展開されている場合は何もしない。
    """

    if KFTT_DIR.exists():
        print(f"すでに展開されています: {KFTT_DIR}")
        return

    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"圧縮ファイルが見つかりません: {ARCHIVE_PATH}\n"
            "Pythonファイルと同じディレクトリに"
            " kftt-data-1.0.tar.gz を置いてください。"
        )

    print(f"展開中: {ARCHIVE_PATH}")

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as tar:
        tar.extractall(path=BASE_DIR)

    print(f"展開完了: {KFTT_DIR}")


def tokenize_japanese(text: str) -> str:
    """
    日本語文を形態素単位に分割する。

    例:
        京都は日本の都市です。
        ↓
        京都 は 日本 の 都市 です 。
    """

    tokens = [token.surface for token in ja_tokenizer.tokenize(text)]
    return " ".join(tokens)


def tokenize_english(text: str) -> str:
    """
    英語文を単語・句読点単位に分割する。

    例:
        Kyoto is a city in Japan.
        ↓
        Kyoto is a city in Japan .
    """

    tokens = en_tokenizer.tokenize(text)
    return " ".join(tokens)


def preprocess_file(
    input_path: Path,
    output_path: Path,
    language: str,
) -> int:
    """
    1つのファイルを読み込み、各行をトークン化して保存する。

    Parameters
    ----------
    input_path:
        前処理する元ファイル
    output_path:
        保存先
    language:
        "ja" または "en"

    Returns
    -------
    int:
        処理した文の数
    """

    if language not in {"ja", "en"}:
        raise ValueError("languageには 'ja' または 'en' を指定してください。")

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentence_count = 0

    with (
        input_path.open("r", encoding="utf-8") as reader,
        output_path.open("w", encoding="utf-8") as writer,
    ):
        for line in reader:
            # 行末の改行と余分な空白を除去
            sentence = line.strip()

            # 空行の場合は空行のまま保存する
            if not sentence:
                writer.write("\n")
                sentence_count += 1
                continue

            if language == "ja":
                tokenized = tokenize_japanese(sentence)
            else:
                tokenized = tokenize_english(sentence)

            writer.write(tokenized + "\n")
            sentence_count += 1

    return sentence_count


def prepare_split(split: str) -> None:
    """
    train、dev、testのいずれかについて、
    日本語と英語の両方を前処理する。
    """

    ja_input = ORIG_DIR / f"kyoto-{split}.ja"
    en_input = ORIG_DIR / f"kyoto-{split}.en"

    ja_output = OUTPUT_DIR / f"{split}.ja"
    en_output = OUTPUT_DIR / f"{split}.en"

    print(f"\n{split}データを処理しています。")

    ja_count = preprocess_file(
        input_path=ja_input,
        output_path=ja_output,
        language="ja",
    )

    en_count = preprocess_file(
        input_path=en_input,
        output_path=en_output,
        language="en",
    )

    # 対訳データなので日英の文数が同じであることを確認
    if ja_count != en_count:
        raise ValueError(
            f"{split}データの日英の文数が一致しません。\n"
            f"日本語: {ja_count}文\n"
            f"英語: {en_count}文"
        )

    print(f"  日本語: {ja_output}")
    print(f"  英語  : {en_output}")
    print(f"  対訳文数: {ja_count}")


def show_examples(split: str = "train", number: int = 3) -> None:
    """
    前処理後の対訳文を先頭から表示する。
    """

    ja_path = OUTPUT_DIR / f"{split}.ja"
    en_path = OUTPUT_DIR / f"{split}.en"

    print(f"\n{split}データの先頭{number}件")

    with (
        ja_path.open("r", encoding="utf-8") as ja_file,
        en_path.open("r", encoding="utf-8") as en_file,
    ):
        for index, (ja_line, en_line) in enumerate(
            zip(ja_file, en_file),
            start=1,
        ):
            if index > number:
                break

            print(f"\n例 {index}")
            print(f"日本語: {ja_line.strip()}")
            print(f"英語  : {en_line.strip()}")


def main() -> None:
    # 1. データセットの展開
    extract_dataset()

    # 2. 出力ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 訓練・開発・評価データを前処理
    for split in ["train", "dev", "test"]:
        prepare_split(split)

    # 4. 前処理結果の確認
    show_examples(split="train", number=3)

    print("\nすべての前処理が完了しました。")
    print(f"保存先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#出力
