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
