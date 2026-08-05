from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import math
from dataclasses import dataclass
from itertools import count, zip_longest
from pathlib import Path

import matplotlib
import sacrebleu
import torch
import torch.nn.functional as F
from nltk.tokenize.treebank import TreebankWordDetokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# パスと設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRANSLATION_PROGRAM_PATH = BASE_DIR / "92_2020ver.py"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "91_transformer"
    / "best_model.pt"
)

DEV_JA_PATH = BASE_DIR / "data" / "processed" / "dev.ja"
DEV_EN_PATH = BASE_DIR / "data" / "processed" / "dev.en"

OUTPUT_DIR = BASE_DIR / "outputs" / "94_beam_search"

SCORE_TABLE_PATH = OUTPUT_DIR / "beam_bleu_scores.tsv"
PLOT_PATH = OUTPUT_DIR / "beam_bleu_plot.png"

DEFAULT_BEAM_SIZES = [1, 2, 5, 10, 20, 50, 100]
LOG_INTERVAL = 100


# ============================================================
# 特殊トークン
# ============================================================

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


english_detokenizer = TreebankWordDetokenizer()


# ============================================================
# 92のプログラムを読み込む
# ============================================================

def load_translation_module():
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
# 開発データの読み込み
# ============================================================

def load_parallel_data(
    source_path: Path,
    reference_path: Path,
    max_examples: int | None,
) -> tuple[list[str], list[str]]:
    if not source_path.exists():
        raise FileNotFoundError(
            f"開発用日本語ファイルが見つかりません: {source_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"開発用英語ファイルが見つかりません: {reference_path}"
        )

    source_sentences: list[str] = []
    reference_sentences: list[str] = []

    with (
        source_path.open("r", encoding="utf-8") as source_file,
        reference_path.open("r", encoding="utf-8") as reference_file,
    ):
        paired_lines = zip_longest(
            source_file,
            reference_file,
            fillvalue=None,
        )

        for line_number, (source_line, reference_line) in enumerate(
            paired_lines,
            start=1,
        ):
            if source_line is None or reference_line is None:
                raise ValueError(
                    "dev.jaとdev.enの行数が一致しません。\n"
                    f"不一致を検出した行: {line_number}"
                )

            source_sentence = source_line.strip()
            reference_sentence = reference_line.strip()

            if not source_sentence or not reference_sentence:
                continue

            source_sentences.append(source_sentence)
            reference_sentences.append(reference_sentence)

            if (
                max_examples is not None
                and len(source_sentences) >= max_examples
            ):
                break

    if not source_sentences:
        raise ValueError("開発データが読み込まれませんでした。")

    return source_sentences, reference_sentences


# ============================================================
# ID変換と出力変換
# ============================================================

def encode_source_sentence(
    sentence: str,
    source_vocab,
    bos_id: int,
    eos_id: int,
    unk_id: int,
    max_length: int,
) -> torch.Tensor:
    tokens = sentence.split()
    tokens = tokens[: max_length - 2]

    token_ids = [
        source_vocab.token_to_id.get(token, unk_id)
        for token in tokens
    ]

    return torch.tensor(
        [bos_id, *token_ids, eos_id],
        dtype=torch.long,
    )


def ids_to_tokens(
    token_ids: list[int],
    target_vocab,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> list[str]:
    tokens: list[str] = []

    for token_id in token_ids:
        if token_id == eos_id:
            break

        if token_id in {pad_id, bos_id}:
            continue

        if 0 <= token_id < len(target_vocab.id_to_token):
            token = target_vocab.id_to_token[token_id]
        else:
            token = "<unk>"

        tokens.append(token)

    return tokens


def detokenize_references(
    tokenized_references: list[str],
) -> list[str]:
    return [
        english_detokenizer.detokenize(sentence.split())
        for sentence in tokenized_references
    ]


# ============================================================
# ビーム探索
# ============================================================

@dataclass(frozen=True)
class BeamHypothesis:
    token_ids: list[int]
    log_prob_sum: float
    generated_length: int

    @property
    def average_score(self) -> float:
        return self.log_prob_sum / max(self.generated_length, 1)


def keep_best(
    hypotheses: list[BeamHypothesis],
    beam_size: int,
) -> list[BeamHypothesis]:
    """
    平均対数確率が高い候補をbeam_size個だけ残す。
    heapqは最小ヒープなので、ヒープの根が現在の最下位候補になる。
    """

    heap: list[tuple[float, int, BeamHypothesis]] = []
    tie_breaker = count()

    for hypothesis in hypotheses:
        item = (
            hypothesis.average_score,
            next(tie_breaker),
            hypothesis,
        )

        if len(heap) < beam_size:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    heap.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        item[2]
        for item in heap
    ]


@torch.inference_mode()
def translate_with_beam_search(
    sentence: str,
    model,
    source_vocab,
    target_vocab,
    device: torch.device,
    pad_id: int,
    unk_id: int,
    bos_id: int,
    eos_id: int,
    max_length: int,
    beam_size: int,
) -> list[str]:
    """
    1文をビーム探索で翻訳する。

    スコアは、生成した各トークンの対数生成確率の平均とする。
    BOSはスコア計算に含めず、EOSは生成確率として含める。
    """

    source_tensor = encode_source_sentence(
        sentence=sentence,
        source_vocab=source_vocab,
        bos_id=bos_id,
        eos_id=eos_id,
        unk_id=unk_id,
        max_length=max_length,
    ).unsqueeze(0).to(device)

    # Transformer encoder output is identical at every decoding step, so
    # compute it once and reuse it for all beam candidates.
    source_padding_mask = source_tensor.eq(pad_id)
    source_embeddings = model.source_embedding(source_tensor)
    source_embeddings *= math.sqrt(model.d_model)
    source_embeddings = model.source_position(source_embeddings)
    memory = model.transformer.encoder(
        source_embeddings,
        src_key_padding_mask=source_padding_mask,
    )

    active_hypotheses = [
        BeamHypothesis(
            token_ids=[bos_id],
            log_prob_sum=0.0,
            generated_length=0,
        )
    ]

    finished_hypotheses: list[BeamHypothesis] = []

    for _ in range(max_length - 1):
        if not active_hypotheses:
            break

        target_batch = torch.tensor(
            [
                hypothesis.token_ids
                for hypothesis in active_hypotheses
            ],
            dtype=torch.long,
            device=device,
        )

        memory_batch = memory.expand(
            len(active_hypotheses),
            -1,
            -1,
        )
        memory_padding_mask = source_padding_mask.expand(
            len(active_hypotheses),
            -1,
        )

        target_padding_mask = target_batch.eq(pad_id)
        target_embeddings = model.target_embedding(target_batch)
        target_embeddings *= math.sqrt(model.d_model)
        target_embeddings = model.target_position(target_embeddings)
        decoder_output = model.transformer.decoder(
            target_embeddings,
            memory_batch,
            tgt_mask=model.create_causal_mask(
                target_length=target_batch.size(1),
                device=device,
            ),
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = model.output_layer(decoder_output)

        next_token_log_probs = F.log_softmax(
            logits[:, -1, :],
            dim=-1,
        )

        next_token_log_probs[:, pad_id] = float("-inf")
        next_token_log_probs[:, bos_id] = float("-inf")

        topk = min(
            beam_size,
            next_token_log_probs.size(-1),
        )

        top_log_probs, top_token_ids = torch.topk(
            next_token_log_probs,
            k=topk,
            dim=-1,
        )

        previous_scores = torch.tensor(
            [hypothesis.log_prob_sum for hypothesis in active_hypotheses],
            dtype=top_log_probs.dtype,
            device=device,
        ).unsqueeze(1)
        cumulative_scores = top_log_probs + previous_scores

        # Add EOS candidates to the finished beam.
        eos_locations = top_token_ids.eq(eos_id).nonzero(as_tuple=False)
        for beam_index, candidate_index in eos_locations.tolist():
            hypothesis = active_hypotheses[beam_index]
            finished_hypotheses.append(
                BeamHypothesis(
                    token_ids=[*hypothesis.token_ids, eos_id],
                    log_prob_sum=float(
                        cumulative_scores[beam_index, candidate_index].item()
                    ),
                    generated_length=hypothesis.generated_length + 1,
                )
            )

        # Select the global top beam_size non-EOS candidates on the GPU.
        active_scores = cumulative_scores.masked_fill(
            top_token_ids.eq(eos_id),
            float("-inf"),
        ).flatten()
        active_topk = min(beam_size, active_scores.numel())
        selected_scores, selected_indices = torch.topk(
            active_scores,
            k=active_topk,
        )
        candidates: list[BeamHypothesis] = []
        for score, flat_index in zip(
            selected_scores.tolist(),
            selected_indices.tolist(),
        ):
            if not math.isfinite(score):
                continue
            beam_index = flat_index // topk
            candidate_index = flat_index % topk
            next_token_id = int(
                top_token_ids[beam_index, candidate_index].item()
            )
            hypothesis = active_hypotheses[beam_index]
            candidates.append(
                BeamHypothesis(
                    token_ids=[*hypothesis.token_ids, next_token_id],
                    log_prob_sum=float(score),
                    generated_length=hypothesis.generated_length + 1,
                )
            )

        active_hypotheses = candidates

        finished_hypotheses = keep_best(
            hypotheses=finished_hypotheses,
            beam_size=beam_size,
        )

    final_hypotheses = (
        finished_hypotheses
        if finished_hypotheses
        else active_hypotheses
    )

    best_hypothesis = max(
        final_hypotheses,
        key=lambda hypothesis: hypothesis.average_score,
    )

    return ids_to_tokens(
        token_ids=best_hypothesis.token_ids[1:],
        target_vocab=target_vocab,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )


# ============================================================
# 評価
# ============================================================

def translate_dev_data(
    source_sentences: list[str],
    model,
    source_vocab,
    target_vocab,
    device: torch.device,
    special_tokens: dict,
    max_length: int,
    beam_size: int,
    log_interval: int,
) -> tuple[list[str], list[str]]:
    pad_id = special_tokens.get("pad_id", PAD_ID)
    unk_id = special_tokens.get("unk_id", UNK_ID)
    bos_id = special_tokens.get("bos_id", BOS_ID)
    eos_id = special_tokens.get("eos_id", EOS_ID)

    tokenized_hypotheses: list[str] = []
    detokenized_hypotheses: list[str] = []

    for index, sentence in enumerate(
        source_sentences,
        start=1,
    ):
        tokens = translate_with_beam_search(
            sentence=sentence,
            model=model,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            device=device,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            max_length=max_length,
            beam_size=beam_size,
        )

        tokenized_sentence = " ".join(tokens)
        detokenized_sentence = (
            english_detokenizer.detokenize(tokens)
        )

        tokenized_hypotheses.append(tokenized_sentence)
        detokenized_hypotheses.append(detokenized_sentence)

        if (
            index % log_interval == 0
            or index == len(source_sentences)
        ):
            print(
                f"beam={beam_size}: "
                f"{index:,}/{len(source_sentences):,}文を翻訳"
            )

    return tokenized_hypotheses, detokenized_hypotheses


def save_lines(
    output_path: Path,
    lines: list[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for line in lines:
            output_file.write(line + "\n")


def save_score_table(
    output_path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "beam_size",
                "bleu",
                "hypothesis_path",
            ],
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    beam_sizes: list[int],
    bleu_scores: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        beam_sizes,
        bleu_scores,
        marker="o",
    )
    plt.xlabel("Beam size")
    plt.ylabel("BLEU")
    plt.title("BLEU on development set with beam search")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(beam_sizes)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_beam_sizes(
    values: list[str],
) -> list[int]:
    beam_sizes: list[int] = []

    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue

            beam_size = int(part)
            if beam_size < 1:
                raise ValueError(
                    "beam sizeは1以上にしてください。"
                )

            beam_sizes.append(beam_size)

    if not beam_sizes:
        raise ValueError("beam sizeが指定されていません。")

    return sorted(set(beam_sizes))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "91で学習したTransformer翻訳モデルを使い、"
            "開発セット上でビーム幅ごとのBLEUを評価します。"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="学習済みモデルのパス",
    )

    parser.add_argument(
        "--beam-sizes",
        nargs="+",
        default=[
            str(size)
            for size in DEFAULT_BEAM_SIZES
        ],
        help=(
            "評価するビーム幅。例: --beam-sizes 1 5 10 20 50 100 "
            "または --beam-sizes 1,5,10,20,50,100"
        ),
    )

    parser.add_argument(
        "--max-dev-examples",
        type=int,
        default=None,
        help="動作確認用。先頭から指定文数だけ評価する。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="出力ディレクトリ",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL,
        help="進捗表示の間隔",
    )

    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="文数が一致する既存の翻訳結果を再利用する。",
    )

    args = parser.parse_args()

    beam_sizes = parse_beam_sizes(args.beam_sizes)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("ビーム探索によるBLEU評価")
    print("=" * 60)
    print(f"使用デバイス: {device}")

    if device.type == "cuda":
        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )
    else:
        print(
            "注意: GPUが利用できないためCPUで実行します。"
        )

    translation_module = load_translation_module()

    print(f"\nモデルを読み込んでいます: {args.model}")

    (
        model,
        source_vocab,
        target_vocab,
        checkpoint,
    ) = translation_module.load_model(
        model_path=args.model,
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
    print(f"日本語語彙数: {len(source_vocab):,}")
    print(f"英語語彙数  : {len(target_vocab):,}")
    print(f"最大生成長  : {max_length}")
    print(f"ビーム幅    : {beam_sizes}")

    print("\n開発データを読み込んでいます。")

    source_sentences, tokenized_references = load_parallel_data(
        source_path=DEV_JA_PATH,
        reference_path=DEV_EN_PATH,
        max_examples=args.max_dev_examples,
    )

    detokenized_references = detokenize_references(
        tokenized_references
    )

    print(f"開発文数: {len(source_sentences):,}")

    bleu_metric = sacrebleu.metrics.BLEU(
        tokenize="13a",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, float | int | str]] = []
    bleu_scores: list[float] = []

    for beam_size in beam_sizes:
        print("\n" + "-" * 60)
        print(f"beam size = {beam_size}")
        print("-" * 60)

        tokenized_output_path = (
            args.output_dir
            / f"dev.hypothesis.beam{beam_size}.tokenized.en"
        )

        detokenized_output_path = (
            args.output_dir
            / f"dev.hypothesis.beam{beam_size}.detokenized.en"
        )

        reused = False
        if args.reuse_existing and detokenized_output_path.exists():
            detokenized_hypotheses = detokenized_output_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if len(detokenized_hypotheses) == len(source_sentences):
                reused = True
                print(f"既存結果を再利用: {detokenized_output_path}")

        if not reused:
            (
                tokenized_hypotheses,
                detokenized_hypotheses,
            ) = translate_dev_data(
                source_sentences=source_sentences,
                model=model,
                source_vocab=source_vocab,
                target_vocab=target_vocab,
                device=device,
                special_tokens=special_tokens,
                max_length=max_length,
                beam_size=beam_size,
                log_interval=args.log_interval,
            )

            save_lines(
                tokenized_output_path,
                tokenized_hypotheses,
            )
            save_lines(
                detokenized_output_path,
                detokenized_hypotheses,
            )

        bleu_result = bleu_metric.corpus_score(
            detokenized_hypotheses,
            [detokenized_references],
        )

        rows.append(
            {
                "beam_size": beam_size,
                "bleu": f"{bleu_result.score:.4f}",
                "hypothesis_path": str(detokenized_output_path),
            }
        )
        bleu_scores.append(bleu_result.score)

        print(f"BLEU: {bleu_result.score:.4f}")
        print(f"詳細: {bleu_result}")

    score_table_path = args.output_dir / SCORE_TABLE_PATH.name
    plot_path = args.output_dir / PLOT_PATH.name

    save_score_table(
        score_table_path,
        rows,
    )
    save_plot(
        beam_sizes,
        bleu_scores,
        plot_path,
    )

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)
    print(f"スコア表: {score_table_path}")
    print(f"プロット: {plot_path}")
    print(f"Signature: {bleu_metric.get_signature()}")


if __name__ == "__main__":
    main()
