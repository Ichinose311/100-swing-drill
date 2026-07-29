from __future__ import annotations

import importlib.util
from itertools import zip_longest
from pathlib import Path

import sacrebleu
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# パスと設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 92で定義したTransformerモデルを再利用する
TRANSLATION_PROGRAM_PATH = BASE_DIR / "92_2020ver.py"

# 91で学習・保存したモデル
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "91_transformer"
    / "best_model.pt"
)

# 90で作成した評価データ
TEST_JA_PATH = BASE_DIR / "data" / "processed" / "test.ja"
TEST_EN_PATH = BASE_DIR / "data" / "processed" / "test.en"

# 翻訳結果の保存先
OUTPUT_DIR = BASE_DIR / "outputs" / "93_bleu"

HYPOTHESIS_TOKENIZED_PATH = (
    OUTPUT_DIR / "test.hypothesis.tokenized.en"
)

HYPOTHESIS_DETOKENIZED_PATH = (
    OUTPUT_DIR / "test.hypothesis.detokenized.en"
)

REFERENCE_DETOKENIZED_PATH = (
    OUTPUT_DIR / "test.reference.detokenized.en"
)

RESULT_PATH = OUTPUT_DIR / "bleu_result.txt"

# 評価時のバッチサイズ
# GPUメモリ不足の場合は32や16に下げる
BATCH_SIZE = 64

# 何バッチごとに進捗を表示するか
LOG_INTERVAL = 10


# ============================================================
# 英語のデトークナイザ
# ============================================================

english_detokenizer = TreebankWordDetokenizer()


# ============================================================
# 92のプログラムを読み込む
# ============================================================

def load_translation_module():
    """
    ファイル名が92_2020ver.pyで数字から始まるため、
    importlibを使って読み込む。
    """

    if not TRANSLATION_PROGRAM_PATH.exists():
        raise FileNotFoundError(
            "92のプログラムが見つかりません。\n"
            f"確認した場所: {TRANSLATION_PROGRAM_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "translation_program",
        TRANSLATION_PROGRAM_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"92のプログラムを読み込めません: "
            f"{TRANSLATION_PROGRAM_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# 評価データの読み込み
# ============================================================

def load_test_data(
    source_path: Path,
    reference_path: Path,
) -> tuple[list[str], list[str]]:
    """
    評価用の日本語文と英語の正解訳を読み込む。

    日本語文と英語文の同じ行番号が対応している。
    """

    if not source_path.exists():
        raise FileNotFoundError(
            f"評価用日本語ファイルが見つかりません: "
            f"{source_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"評価用英語ファイルが見つかりません: "
            f"{reference_path}"
        )

    source_sentences: list[str] = []
    reference_sentences: list[str] = []

    with (
        source_path.open(
            "r",
            encoding="utf-8",
        ) as source_file,
        reference_path.open(
            "r",
            encoding="utf-8",
        ) as reference_file,
    ):
        paired_lines = zip_longest(
            source_file,
            reference_file,
            fillvalue=None,
        )

        for line_number, (
            source_line,
            reference_line,
        ) in enumerate(paired_lines, start=1):

            if source_line is None or reference_line is None:
                raise ValueError(
                    "test.jaとtest.enの行数が一致しません。\n"
                    f"不一致を検出した行: {line_number}"
                )

            source_sentence = source_line.strip()
            reference_sentence = reference_line.strip()

            # 空の対訳文は評価対象から除外する
            if not source_sentence or not reference_sentence:
                continue

            source_sentences.append(source_sentence)
            reference_sentences.append(reference_sentence)

    if not source_sentences:
        raise ValueError(
            "評価データが読み込まれませんでした。"
        )

    return source_sentences, reference_sentences


# ============================================================
# 日本語文をIDに変換
# ============================================================

def encode_source_sentence(
    sentence: str,
    source_vocab,
    bos_id: int,
    eos_id: int,
    unk_id: int,
    max_length: int,
) -> torch.Tensor:
    """
    90で既に形態素分割されている日本語文をID列にする。

    例:
        京都 に は 多く の 寺 が あり ます 。

        ↓

        <bos> 京都 に は 多く の 寺 が あり ます 。 <eos>
    """

    tokens = sentence.split()

    # BOSとEOSの分を空ける
    tokens = tokens[: max_length - 2]

    token_ids = [
        source_vocab.token_to_id.get(
            token,
            unk_id,
        )
        for token in tokens
    ]

    source_ids = [
        bos_id,
        *token_ids,
        eos_id,
    ]

    return torch.tensor(
        source_ids,
        dtype=torch.long,
    )


# ============================================================
# バッチ単位の貪欲法
# ============================================================

@torch.inference_mode()
def translate_batch(
    source_sentences: list[str],
    model,
    source_vocab,
    target_vocab,
    device: torch.device,
    pad_id: int,
    unk_id: int,
    bos_id: int,
    eos_id: int,
    max_length: int,
) -> list[list[str]]:
    """
    複数の日本語文を一度に翻訳する。

    各生成位置において、最も確率が高い英語トークンを
    選択する貪欲法を使用する。
    """

    encoded_sentences = [
        encode_source_sentence(
            sentence=sentence,
            source_vocab=source_vocab,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=unk_id,
            max_length=max_length,
        )
        for sentence in source_sentences
    ]

    # バッチ内の文長をPADでそろえる
    source_batch = pad_sequence(
        encoded_sentences,
        batch_first=True,
        padding_value=pad_id,
    ).to(device)

    batch_size = source_batch.size(0)

    # 最初は全ての文がBOSだけを持つ
    generated_ids = torch.full(
        size=(batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=device,
    )

    # EOSを生成した文を記録する
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    for _ in range(max_length - 1):
        logits = model(
            source=source_batch,
            target=generated_ids,
        )

        # 各文について、最後の位置の予測だけを使用する
        next_token_logits = logits[:, -1, :].clone()

        # PADとBOSは翻訳結果として生成させない
        next_token_logits[:, pad_id] = float("-inf")
        next_token_logits[:, bos_id] = float("-inf")

        next_token_ids = torch.argmax(
            next_token_logits,
            dim=-1,
        )

        # 既に終了した文はEOSを追加し続ける
        next_token_ids = torch.where(
            finished,
            torch.full_like(
                next_token_ids,
                eos_id,
            ),
            next_token_ids,
        )

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token_ids.unsqueeze(1),
            ],
            dim=1,
        )

        finished = finished | next_token_ids.eq(
            eos_id
        )

        # バッチ内の全ての文が終了したら打ち切る
        if bool(finished.all()):
            break

    generated_ids = generated_ids.cpu().tolist()

    translated_batches: list[list[str]] = []

    for sentence_ids in generated_ids:
        translated_tokens: list[str] = []

        # 先頭のBOSを除いて確認する
        for token_id in sentence_ids[1:]:
            if token_id == eos_id:
                break

            if token_id in {pad_id, bos_id}:
                continue

            if 0 <= token_id < len(target_vocab.id_to_token):
                token = target_vocab.id_to_token[token_id]
            else:
                token = "<unk>"

            translated_tokens.append(token)

        translated_batches.append(
            translated_tokens
        )

    return translated_batches


# ============================================================
# 評価データ全体の翻訳
# ============================================================

def translate_test_data(
    source_sentences: list[str],
    model,
    source_vocab,
    target_vocab,
    device: torch.device,
    special_tokens: dict,
    max_length: int,
) -> tuple[list[str], list[str]]:
    """
    評価データをバッチに分けて翻訳する。

    Returns
    -------
    tokenized_hypotheses:
        空白区切りの翻訳結果

    detokenized_hypotheses:
        BLEU計算用にデトークナイズした翻訳結果
    """

    pad_id = special_tokens.get("pad_id", 0)
    unk_id = special_tokens.get("unk_id", 1)
    bos_id = special_tokens.get("bos_id", 2)
    eos_id = special_tokens.get("eos_id", 3)

    tokenized_hypotheses: list[str] = []
    detokenized_hypotheses: list[str] = []

    total_batches = (
        len(source_sentences)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for batch_number, start_index in enumerate(
        range(
            0,
            len(source_sentences),
            BATCH_SIZE,
        ),
        start=1,
    ):
        batch_sentences = source_sentences[
            start_index : start_index + BATCH_SIZE
        ]

        batch_translations = translate_batch(
            source_sentences=batch_sentences,
            model=model,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            device=device,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            max_length=max_length,
        )

        for tokens in batch_translations:
            tokenized_sentence = " ".join(tokens)

            detokenized_sentence = (
                english_detokenizer.detokenize(
                    tokens
                )
            )

            tokenized_hypotheses.append(
                tokenized_sentence
            )

            detokenized_hypotheses.append(
                detokenized_sentence
            )

        if (
            batch_number % LOG_INTERVAL == 0
            or batch_number == total_batches
        ):
            translated_count = min(
                start_index + BATCH_SIZE,
                len(source_sentences),
            )

            print(
                f"翻訳中: "
                f"{translated_count:,}/"
                f"{len(source_sentences):,}文 "
                f"({batch_number}/{total_batches}バッチ)"
            )

    return (
        tokenized_hypotheses,
        detokenized_hypotheses,
    )


# ============================================================
# 正解訳のデトークナイズ
# ============================================================

def detokenize_references(
    tokenized_references: list[str],
) -> list[str]:
    """
    90で単語分割された正解訳を、
    sacreBLEUへ渡す前に英文へ戻す。
    """

    detokenized_references: list[str] = []

    for sentence in tokenized_references:
        tokens = sentence.split()

        detokenized_sentence = (
            english_detokenizer.detokenize(
                tokens
            )
        )

        detokenized_references.append(
            detokenized_sentence
        )

    return detokenized_references


# ============================================================
# ファイル保存
# ============================================================

def save_lines(
    output_path: Path,
    lines: list[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for line in lines:
            output_file.write(line + "\n")


# ============================================================
# 翻訳例の表示
# ============================================================

def show_examples(
    source_sentences: list[str],
    references: list[str],
    hypotheses: list[str],
    number: int = 5,
) -> None:
    print("\n翻訳例")

    display_count = min(
        number,
        len(source_sentences),
    )

    for index in range(display_count):
        print("\n" + "-" * 60)
        print(f"例 {index + 1}")
        print(f"日本語: {source_sentences[index]}")
        print(f"正解訳: {references[index]}")
        print(f"生成訳: {hypotheses[index]}")

    print("-" * 60)


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # デバイス
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("BLEUスコアの計測")
    print("=" * 60)
    print(f"使用デバイス: {device}")

    if device.type == "cuda":
        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )
    else:
        print(
            "注意: GPUが利用できないため、"
            "CPUで翻訳します。"
        )

    # --------------------------------------------------------
    # 92のプログラムを読み込む
    # --------------------------------------------------------

    translation_module = (
        load_translation_module()
    )

    # --------------------------------------------------------
    # 学習済みモデルを読み込む
    # --------------------------------------------------------

    print(f"\nモデルを読み込んでいます: {MODEL_PATH}")

    (
        model,
        source_vocab,
        target_vocab,
        checkpoint,
    ) = translation_module.load_model(
        model_path=MODEL_PATH,
        device=device,
    )

    model.eval()

    model_config = checkpoint["model_config"]
    special_tokens = checkpoint.get(
        "special_tokens",
        {},
    )

    max_length = model_config["max_length"]

    print(
        f"学習済みEpoch: "
        f"{checkpoint.get('epoch', '不明')}"
    )

    print(
        f"日本語語彙数: "
        f"{len(source_vocab):,}"
    )

    print(
        f"英語語彙数  : "
        f"{len(target_vocab):,}"
    )

    # --------------------------------------------------------
    # 評価データの読み込み
    # --------------------------------------------------------

    print("\n評価データを読み込んでいます。")

    (
        source_sentences,
        tokenized_references,
    ) = load_test_data(
        source_path=TEST_JA_PATH,
        reference_path=TEST_EN_PATH,
    )

    print(
        f"評価文数: {len(source_sentences):,}"
    )

    # --------------------------------------------------------
    # 評価データを翻訳
    # --------------------------------------------------------

    print("\n評価データを翻訳しています。")

    (
        tokenized_hypotheses,
        detokenized_hypotheses,
    ) = translate_test_data(
        source_sentences=source_sentences,
        model=model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        device=device,
        special_tokens=special_tokens,
        max_length=max_length,
    )

    # --------------------------------------------------------
    # 正解訳をデトークナイズ
    # --------------------------------------------------------

    detokenized_references = (
        detokenize_references(
            tokenized_references
        )
    )

    if (
        len(detokenized_hypotheses)
        != len(detokenized_references)
    ):
        raise ValueError(
            "生成訳と正解訳の文数が一致しません。\n"
            f"生成訳: {len(detokenized_hypotheses)}\n"
            f"正解訳: {len(detokenized_references)}"
        )

    # --------------------------------------------------------
    # sacreBLEUの計算
    # --------------------------------------------------------

    # 英語用の標準的な13aトークナイザを使用する
    bleu_metric = sacrebleu.metrics.BLEU(
        tokenize="13a",
    )

    bleu_result = bleu_metric.corpus_score(
        detokenized_hypotheses,
        [detokenized_references],
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("評価結果")
    print("=" * 60)
    print(f"BLEUスコア: {bleu_result.score:.2f}")
    print(f"詳細       : {bleu_result}")
    print(f"Signature  : {bleu_metric.get_signature()}")
    print("=" * 60)

    show_examples(
        source_sentences=source_sentences,
        references=detokenized_references,
        hypotheses=detokenized_hypotheses,
        number=5,
    )

    # --------------------------------------------------------
    # 結果保存
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_lines(
        HYPOTHESIS_TOKENIZED_PATH,
        tokenized_hypotheses,
    )

    save_lines(
        HYPOTHESIS_DETOKENIZED_PATH,
        detokenized_hypotheses,
    )

    save_lines(
        REFERENCE_DETOKENIZED_PATH,
        detokenized_references,
    )

    result_text = (
        f"評価文数: {len(source_sentences)}\n"
        f"BLEUスコア: {bleu_result.score:.4f}\n"
        f"詳細: {bleu_result}\n"
        f"Signature: {bleu_metric.get_signature()}\n"
        f"モデル: {MODEL_PATH}\n"
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as result_file:
        result_file.write(result_text)

    print("\n評価結果を保存しました。")
    print(
        f"トークン化生成訳: "
        f"{HYPOTHESIS_TOKENIZED_PATH}"
    )
    print(
        f"デトークナイズ生成訳: "
        f"{HYPOTHESIS_DETOKENIZED_PATH}"
    )
    print(
        f"デトークナイズ正解訳: "
        f"{REFERENCE_DETOKENIZED_PATH}"
    )
    print(f"BLEU評価結果: {RESULT_PATH}")


if __name__ == "__main__":
    main()

#出力
