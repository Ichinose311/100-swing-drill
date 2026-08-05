from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path

import sacrebleu
from nltk.tokenize.treebank import TreebankWordDetokenizer


BASE_DIR = Path(__file__).resolve().parent
TASK95_PATH = BASE_DIR / "95_2020ver.py"
DATA_DIR = BASE_DIR / "data" / "processed"
ORIG_DIR = BASE_DIR / "kftt-data-1.0" / "data" / "orig"
DOMAIN_DIR = BASE_DIR / "data" / "domain_adaptation"
MODEL_DIR = BASE_DIR / "models" / "98_domain_adapt"
OUTPUT_DIR = BASE_DIR / "outputs" / "98_domain_adapt"
DEFAULT_BASELINE = BASE_DIR / "models" / "95_sentencepiece" / "best_model.pt"
DETOKENIZER = TreebankWordDetokenizer()


def read_kftt(max_examples: int | None = None) -> list[tuple[str, str]]:
    pairs = []
    with (
        (ORIG_DIR / "kyoto-train.ja").open(encoding="utf-8") as ja_file,
        (ORIG_DIR / "kyoto-train.en").open(encoding="utf-8") as en_file,
    ):
        for ja_line, en_line in zip_longest(ja_file, en_file):
            if ja_line is None or en_line is None:
                raise ValueError("KFTT train.ja and train.en have different line counts")
            ja, en = ja_line.strip(), en_line.strip()
            if ja and en:
                pairs.append((ja, en))
            if max_examples is not None and len(pairs) >= max_examples:
                break
    return pairs


def read_jesc(
    input_path: Path,
    english_column: int,
    japanese_column: int,
    max_examples: int | None,
    max_tokens: int,
) -> list[tuple[str, str]]:
    pairs = []
    largest_column = max(english_column, japanese_column)
    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) <= largest_column:
                if line_number <= 5:
                    print(f"Skip malformed JESC line {line_number}")
                continue
            en = DETOKENIZER.detokenize(columns[english_column].strip().split())
            # The official JESC split is tokenized. KFTT SentencePiece is trained
            # on original Japanese, so remove tokenizer-inserted spaces here.
            ja = "".join(columns[japanese_column].strip().split())
            if not ja or not en:
                continue
            if len(ja.split()) > max_tokens or len(en.split()) > max_tokens:
                continue
            pairs.append((ja, en))
            if max_examples is not None and len(pairs) >= max_examples:
                break
    if not pairs:
        raise ValueError(f"No JESC sentence pairs were read from {input_path}")
    return pairs


def write_parallel(pairs: list[tuple[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.ja").write_text(
        "\n".join(ja for ja, _ in pairs) + "\n", encoding="utf-8"
    )
    (output_dir / "train.en").write_text(
        "\n".join(en for _, en in pairs) + "\n", encoding="utf-8"
    )


def prepare(args: argparse.Namespace) -> None:
    kftt = read_kftt(args.max_kftt_examples)
    requested_external = int(len(kftt) * args.external_ratio)
    if args.max_external_examples is not None:
        requested_external = min(requested_external, args.max_external_examples)
    jesc = read_jesc(
        Path(args.jesc_train),
        args.english_column,
        args.japanese_column,
        requested_external,
        args.max_tokens,
    )
    pairs = [*kftt, *jesc]
    random.Random(args.seed).shuffle(pairs)
    write_parallel(pairs, Path(args.mixed_dir))
    metadata = {
        "kftt_pairs": len(kftt),
        "jesc_pairs": len(jesc),
        "mixed_pairs": len(pairs),
        "seed": args.seed,
        "jesc_source": str(Path(args.jesc_train).resolve()),
    }
    (Path(args.mixed_dir) / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def train(args: argparse.Namespace) -> None:
    mixed_dir = Path(args.mixed_dir)
    if not (mixed_dir / "train.ja").exists():
        raise FileNotFoundError("Run the prepare subcommand before train.")
    command = [
        sys.executable,
        str(TASK95_PATH),
        "train",
        "--train-ja",
        str(mixed_dir / "train.ja"),
        "--train-en",
        str(mixed_dir / "train.en"),
        "--model-dir",
        str(args.model_dir),
        "--init-checkpoint",
        str(args.baseline_checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer",
        args.optimizer,
        "--warmup-steps",
        str(args.warmup_steps),
        "--bleu-train-samples",
        str(args.bleu_train_samples),
        "--run-name",
        "jesc_mixed_finetuning",
        "--tensorboard",
    ]
    subprocess.run(command, cwd=BASE_DIR, check=True)


def run_eval(checkpoint: Path, output: Path, beam_size: int) -> float:
    command = [
        sys.executable,
        str(TASK95_PATH),
        "eval",
        "--checkpoint",
        str(checkpoint),
        "--source",
        str(ORIG_DIR / "kyoto-test.ja"),
        "--reference",
        str(ORIG_DIR / "kyoto-test.en"),
        "--output",
        str(output),
        "--beam-size",
        str(beam_size),
    ]
    subprocess.run(command, cwd=BASE_DIR, check=True)
    hypotheses = output.read_text(encoding="utf-8").splitlines()
    references = (ORIG_DIR / "kyoto-test.en").read_text(
        encoding="utf-8"
    ).splitlines()
    return sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a").score


def evaluate(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_bleu = run_eval(
        Path(args.baseline_checkpoint),
        OUTPUT_DIR / "baseline.test.en",
        args.beam_size,
    )
    adapted_checkpoint = Path(args.model_dir) / "best_model.pt"
    adapted_bleu = run_eval(
        adapted_checkpoint,
        OUTPUT_DIR / "adapted.test.en",
        args.beam_size,
    )
    result = {
        "beam_size": args.beam_size,
        "baseline_bleu": baseline_bleu,
        "adapted_bleu": adapted_bleu,
        "improvement": adapted_bleu - baseline_bleu,
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "adapted_checkpoint": str(adapted_checkpoint.resolve()),
    }
    (OUTPUT_DIR / "domain_adaptation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jesc-train", type=Path, required=True)
    parser.add_argument("--english-column", type=int, default=0)
    parser.add_argument("--japanese-column", type=int, default=1)
    parser.add_argument("--external-ratio", type=float, default=1.0)
    parser.add_argument("--max-external-examples", type=int, default=500000)
    parser.add_argument("--max-kftt-examples", type=int)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-dir", type=Path, default=DOMAIN_DIR)


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mixed-dir", type=Path, default=DOMAIN_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "radam"], default="adamw"
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--bleu-train-samples", type=int, default=1000)


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--beam-size", type=int, default=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="98: JESC domain adaptation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    add_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(function=prepare)
    train_parser = subparsers.add_parser("train")
    add_train_arguments(train_parser)
    train_parser.set_defaults(function=train)
    eval_parser = subparsers.add_parser("evaluate")
    add_eval_arguments(eval_parser)
    eval_parser.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()


# ===== EXECUTION RESULT =====
# Log: outputs/98_domain_adapt/full_run.log
# Epoch 03 | 04400/13759 | loss 3.6171 | ppl 37.23 | lr 3.754814e-06 | 153.2s
# Epoch 03 | 04500/13759 | loss 3.6171 | ppl 37.23 | lr 3.748946e-06 | 156.7s
# Epoch 03 | 04600/13759 | loss 3.6175 | ppl 37.24 | lr 3.743105e-06 | 160.1s
# Epoch 03 | 04700/13759 | loss 3.6172 | ppl 37.23 | lr 3.737291e-06 | 163.6s
# Epoch 03 | 04800/13759 | loss 3.6172 | ppl 37.23 | lr 3.731505e-06 | 167.1s
# Epoch 03 | 04900/13759 | loss 3.6170 | ppl 37.23 | lr 3.725745e-06 | 170.5s
# Epoch 03 | 05000/13759 | loss 3.6172 | ppl 37.23 | lr 3.720012e-06 | 174.0s
# Epoch 03 | 05100/13759 | loss 3.6171 | ppl 37.23 | lr 3.714305e-06 | 177.5s
# Epoch 03 | 05200/13759 | loss 3.6175 | ppl 37.25 | lr 3.708625e-06 | 181.0s
# Epoch 03 | 05300/13759 | loss 3.6173 | ppl 37.24 | lr 3.70297e-06 | 184.4s
# Epoch 03 | 05400/13759 | loss 3.6170 | ppl 37.23 | lr 3.697341e-06 | 187.9s
# Epoch 03 | 05500/13759 | loss 3.6169 | ppl 37.22 | lr 3.691738e-06 | 191.4s
# Epoch 03 | 05600/13759 | loss 3.6169 | ppl 37.22 | lr 3.68616e-06 | 194.9s
# Epoch 03 | 05700/13759 | loss 3.6169 | ppl 37.22 | lr 3.680608e-06 | 198.4s
# Epoch 03 | 05800/13759 | loss 3.6171 | ppl 37.23 | lr 3.67508e-06 | 201.9s
# Epoch 03 | 05900/13759 | loss 3.6169 | ppl 37.22 | lr 3.669577e-06 | 205.5s
# Epoch 03 | 06000/13759 | loss 3.6169 | ppl 37.22 | lr 3.664099e-06 | 208.9s
# Epoch 03 | 06100/13759 | loss 3.6168 | ppl 37.22 | lr 3.658645e-06 | 212.4s
# Epoch 03 | 06200/13759 | loss 3.6169 | ppl 37.22 | lr 3.653216e-06 | 215.9s
# Epoch 03 | 06300/13759 | loss 3.6175 | ppl 37.24 | lr 3.647811e-06 | 219.3s
# Epoch 03 | 06400/13759 | loss 3.6172 | ppl 37.23 | lr 3.642429e-06 | 222.8s
# Epoch 03 | 06500/13759 | loss 3.6169 | ppl 37.22 | lr 3.637072e-06 | 226.3s
# Epoch 03 | 06600/13759 | loss 3.6170 | ppl 37.22 | lr 3.631738e-06 | 229.9s
# Epoch 03 | 06700/13759 | loss 3.6169 | ppl 37.22 | lr 3.626427e-06 | 233.4s
# Epoch 03 | 06800/13759 | loss 3.6170 | ppl 37.23 | lr 3.62114e-06 | 236.9s
# Epoch 03 | 06900/13759 | loss 3.6167 | ppl 37.21 | lr 3.615875e-06 | 240.3s
# Epoch 03 | 07000/13759 | loss 3.6167 | ppl 37.21 | lr 3.610634e-06 | 243.8s
# Epoch 03 | 07100/13759 | loss 3.6166 | ppl 37.21 | lr 3.605415e-06 | 247.3s
# Epoch 03 | 07200/13759 | loss 3.6167 | ppl 37.21 | lr 3.600219e-06 | 250.8s
# Epoch 03 | 07300/13759 | loss 3.6171 | ppl 37.23 | lr 3.595045e-06 | 254.2s
# Epoch 03 | 07400/13759 | loss 3.6170 | ppl 37.23 | lr 3.589894e-06 | 257.7s
# Epoch 03 | 07500/13759 | loss 3.6168 | ppl 37.22 | lr 3.584764e-06 | 261.2s
# Epoch 03 | 07600/13759 | loss 3.6168 | ppl 37.22 | lr 3.579657e-06 | 264.6s
# Epoch 03 | 07700/13759 | loss 3.6170 | ppl 37.22 | lr 3.574571e-06 | 268.0s
# Epoch 03 | 07800/13759 | loss 3.6167 | ppl 37.21 | lr 3.569507e-06 | 271.4s
# Epoch 03 | 07900/13759 | loss 3.6164 | ppl 37.20 | lr 3.564464e-06 | 274.9s
# Epoch 03 | 08000/13759 | loss 3.6166 | ppl 37.21 | lr 3.559443e-06 | 278.4s
# Epoch 03 | 08100/13759 | loss 3.6163 | ppl 37.20 | lr 3.554442e-06 | 281.9s
# Epoch 03 | 08200/13759 | loss 3.6163 | ppl 37.20 | lr 3.549463e-06 | 285.3s
# Epoch 03 | 08300/13759 | loss 3.6162 | ppl 37.20 | lr 3.544505e-06 | 288.8s
# Epoch 03 | 08400/13759 | loss 3.6164 | ppl 37.20 | lr 3.539567e-06 | 292.3s
# Epoch 03 | 08500/13759 | loss 3.6163 | ppl 37.20 | lr 3.53465e-06 | 295.7s
# Epoch 03 | 08600/13759 | loss 3.6162 | ppl 37.20 | lr 3.529754e-06 | 299.0s
# Epoch 03 | 08700/13759 | loss 3.6162 | ppl 37.20 | lr 3.524877e-06 | 302.4s
# Epoch 03 | 08800/13759 | loss 3.6161 | ppl 37.19 | lr 3.520021e-06 | 305.8s
# Epoch 03 | 08900/13759 | loss 3.6163 | ppl 37.20 | lr 3.515185e-06 | 309.3s
# Epoch 03 | 09000/13759 | loss 3.6164 | ppl 37.20 | lr 3.510369e-06 | 312.7s
# Epoch 03 | 09100/13759 | loss 3.6165 | ppl 37.21 | lr 3.505572e-06 | 316.3s
# Epoch 03 | 09200/13759 | loss 3.6164 | ppl 37.20 | lr 3.500796e-06 | 319.8s
# Epoch 03 | 09300/13759 | loss 3.6163 | ppl 37.20 | lr 3.496038e-06 | 323.3s
# Epoch 03 | 09400/13759 | loss 3.6161 | ppl 37.19 | lr 3.4913e-06 | 326.7s
# Epoch 03 | 09500/13759 | loss 3.6161 | ppl 37.19 | lr 3.486581e-06 | 330.2s
# Epoch 03 | 09600/13759 | loss 3.6161 | ppl 37.19 | lr 3.481881e-06 | 333.7s
# Epoch 03 | 09700/13759 | loss 3.6161 | ppl 37.19 | lr 3.477201e-06 | 337.2s
# Epoch 03 | 09800/13759 | loss 3.6161 | ppl 37.19 | lr 3.472539e-06 | 340.6s
# Epoch 03 | 09900/13759 | loss 3.6160 | ppl 37.19 | lr 3.467895e-06 | 344.1s
# Epoch 03 | 10000/13759 | loss 3.6162 | ppl 37.20 | lr 3.463271e-06 | 347.6s
# Epoch 03 | 10100/13759 | loss 3.6160 | ppl 37.19 | lr 3.458664e-06 | 351.1s
# Epoch 03 | 10200/13759 | loss 3.6161 | ppl 37.19 | lr 3.454076e-06 | 354.5s
# Epoch 03 | 10300/13759 | loss 3.6160 | ppl 37.19 | lr 3.449507e-06 | 358.0s
# Epoch 03 | 10400/13759 | loss 3.6159 | ppl 37.19 | lr 3.444955e-06 | 361.5s
# Epoch 03 | 10500/13759 | loss 3.6161 | ppl 37.19 | lr 3.440421e-06 | 365.1s
# Epoch 03 | 10600/13759 | loss 3.6161 | ppl 37.19 | lr 3.435905e-06 | 368.6s
# Epoch 03 | 10700/13759 | loss 3.6162 | ppl 37.19 | lr 3.431407e-06 | 372.0s
# Epoch 03 | 10800/13759 | loss 3.6160 | ppl 37.19 | lr 3.426927e-06 | 375.5s
# Epoch 03 | 10900/13759 | loss 3.6158 | ppl 37.18 | lr 3.422464e-06 | 378.9s
# Epoch 03 | 11000/13759 | loss 3.6158 | ppl 37.18 | lr 3.418018e-06 | 382.4s
# Epoch 03 | 11100/13759 | loss 3.6156 | ppl 37.17 | lr 3.41359e-06 | 385.9s
# Epoch 03 | 11200/13759 | loss 3.6158 | ppl 37.18 | lr 3.409179e-06 | 389.4s
# Epoch 03 | 11300/13759 | loss 3.6158 | ppl 37.18 | lr 3.404785e-06 | 392.9s
# Epoch 03 | 11400/13759 | loss 3.6157 | ppl 37.18 | lr 3.400408e-06 | 396.4s
# Epoch 03 | 11500/13759 | loss 3.6155 | ppl 37.17 | lr 3.396047e-06 | 399.8s
# Epoch 03 | 11600/13759 | loss 3.6157 | ppl 37.18 | lr 3.391704e-06 | 403.3s
# Epoch 03 | 11700/13759 | loss 3.6158 | ppl 37.18 | lr 3.387377e-06 | 406.7s
# Epoch 03 | 11800/13759 | loss 3.6158 | ppl 37.18 | lr 3.383067e-06 | 410.1s
# Epoch 03 | 11900/13759 | loss 3.6158 | ppl 37.18 | lr 3.378773e-06 | 413.6s
# Epoch 03 | 12000/13759 | loss 3.6158 | ppl 37.18 | lr 3.374495e-06 | 417.1s
# Epoch 03 | 12100/13759 | loss 3.6158 | ppl 37.18 | lr 3.370233e-06 | 420.6s
# Epoch 03 | 12200/13759 | loss 3.6158 | ppl 37.18 | lr 3.365988e-06 | 424.1s
# Epoch 03 | 12300/13759 | loss 3.6157 | ppl 37.18 | lr 3.361759e-06 | 427.5s
# Epoch 03 | 12400/13759 | loss 3.6158 | ppl 37.18 | lr 3.357545e-06 | 431.1s
# Epoch 03 | 12500/13759 | loss 3.6158 | ppl 37.18 | lr 3.353348e-06 | 434.6s
# Epoch 03 | 12600/13759 | loss 3.6156 | ppl 37.18 | lr 3.349166e-06 | 438.1s
# Epoch 03 | 12700/13759 | loss 3.6156 | ppl 37.17 | lr 3.344999e-06 | 441.6s
# Epoch 03 | 12800/13759 | loss 3.6156 | ppl 37.17 | lr 3.340848e-06 | 445.0s
# Epoch 03 | 12900/13759 | loss 3.6156 | ppl 37.17 | lr 3.336713e-06 | 448.4s
# Epoch 03 | 13000/13759 | loss 3.6155 | ppl 37.17 | lr 3.332593e-06 | 451.9s
# Epoch 03 | 13100/13759 | loss 3.6154 | ppl 37.17 | lr 3.328488e-06 | 455.3s
# Epoch 03 | 13200/13759 | loss 3.6153 | ppl 37.16 | lr 3.324398e-06 | 458.9s
# Epoch 03 | 13300/13759 | loss 3.6153 | ppl 37.16 | lr 3.320323e-06 | 462.3s
# Epoch 03 | 13400/13759 | loss 3.6153 | ppl 37.16 | lr 3.316264e-06 | 465.8s
# Epoch 03 | 13500/13759 | loss 3.6152 | ppl 37.16 | lr 3.312219e-06 | 469.2s
# Epoch 03 | 13600/13759 | loss 3.6151 | ppl 37.16 | lr 3.308189e-06 | 472.7s
# Epoch 03 | 13700/13759 | loss 3.6150 | ppl 37.15 | lr 3.304173e-06 | 476.1s
# Epoch 03 complete | train loss 3.6150 | dev loss 3.4666 | train BLEU 15.55 | dev BLEU 17.14 | 484.5s
# {
#   "best_epoch": 1,
#   "best_dev_bleu": 17.33971359341473,
#   "best_dev_loss": 3.466916470935973,
#   "best_model": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/98_domain_adapt/best_model.pt",
#   "batch_size": 64,
#   "learning_rate": 3e-05,
#   "optimizer": "adamw"
# }
# /data/student/i2311021/conda_envs/100-swing-drill-gpu/lib/python3.10/site-packages/torch/nn/modules/transformer.py:531: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
#   output = torch._nested_tensor_from_mask(
# BLEU = 19.6064
# Hypotheses: /data/student/i2311021/projects/100-swing-drill/chapter_10_2020/outputs/98_domain_adapt/baseline.test.en
# /data/student/i2311021/conda_envs/100-swing-drill-gpu/lib/python3.10/site-packages/torch/nn/modules/transformer.py:531: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
#   output = torch._nested_tensor_from_mask(
# BLEU = 20.5523
# Hypotheses: /data/student/i2311021/projects/100-swing-drill/chapter_10_2020/outputs/98_domain_adapt/adapted.test.en
# {
#   "beam_size": 5,
#   "baseline_bleu": 19.606396888108463,
#   "adapted_bleu": 20.552280741904887,
#   "improvement": 0.945883853796424,
#   "baseline_checkpoint": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/97_tuning/bs64_lr1em03_adamw/best_model.pt",
#   "adapted_checkpoint": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/98_domain_adapt/best_model.pt"
# }
