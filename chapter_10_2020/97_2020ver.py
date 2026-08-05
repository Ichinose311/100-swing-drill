from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
TASK95_PATH = BASE_DIR / "95_2020ver.py"
MODEL_ROOT = BASE_DIR / "models" / "97_tuning"
OUTPUT_DIR = BASE_DIR / "outputs" / "97_tuning"


def select_trials(max_trials: int, seed: int) -> list[tuple[int, float, str]]:
    batch_sizes = [32, 64, 128]
    learning_rates = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
    optimizers = ["adam", "adamw", "radam"]
    anchors = [
        (64, 3e-4, "adamw"),
        (64, 1e-4, "adam"),
        (64, 1e-4, "radam"),
        (32, 3e-4, "adamw"),
        (128, 3e-4, "adamw"),
        (64, 1e-3, "adamw"),
        (64, 1e-5, "adamw"),
        (64, 1e-6, "adamw"),
    ]
    grid = list(itertools.product(batch_sizes, learning_rates, optimizers))
    remaining = [trial for trial in grid if trial not in anchors]
    random.Random(seed).shuffle(remaining)
    return (anchors + remaining)[:max_trials]


def trial_name(batch_size: int, learning_rate: float, optimizer: str) -> str:
    lr_text = f"{learning_rate:.0e}".replace("-", "m")
    return f"bs{batch_size}_lr{lr_text}_{optimizer}"


def run_trial(
    args: argparse.Namespace,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
) -> dict:
    name = trial_name(batch_size, learning_rate, optimizer)
    model_dir = Path(args.model_root) / name
    summary_path = model_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"Reuse completed trial: {name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        command = [
            sys.executable,
            str(TASK95_PATH),
            "train",
            "--model-dir",
            str(model_dir),
            "--run-name",
            name,
            "--log-dir",
            str(Path(args.output_dir) / "tensorboard"),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--optimizer",
            optimizer,
            "--epochs",
            str(args.epochs),
            "--bleu-train-samples",
            str(args.bleu_train_samples),
            "--seed",
            str(args.seed),
        ]
        if not args.no_tensorboard:
            command.append("--tensorboard")
        if args.max_train_examples is not None:
            command.extend(["--max-train-examples", str(args.max_train_examples)])
        if args.max_dev_examples is not None:
            command.extend(["--max-dev-examples", str(args.max_dev_examples)])
        print("\n" + "=" * 72)
        print(f"Trial: {name}")
        print("=" * 72)
        subprocess.run(command, cwd=BASE_DIR, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "trial": name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "best_epoch": summary["best_epoch"],
        "best_dev_bleu": summary["best_dev_bleu"],
        "best_dev_loss": summary["best_dev_loss"],
        "best_model": summary["best_model"],
    }


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda row: row["best_dev_bleu"], reverse=True)
    with (output_dir / "tuning_results.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=ranked[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(ranked)
    (output_dir / "best_trial.json").write_text(
        json.dumps(ranked[0], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    colors = {"adam": "#2676b8", "adamw": "#d1495b", "radam": "#2a9d6f"}
    markers = {32: "o", 64: "s", 128: "^"}
    plt.figure(figsize=(9, 5.5))
    used_labels: set[str] = set()
    for row in ranked:
        label = row["optimizer"]
        plot_label = label if label not in used_labels else None
        used_labels.add(label)
        plt.scatter(
            row["learning_rate"],
            row["best_dev_bleu"],
            color=colors[row["optimizer"]],
            marker=markers[row["batch_size"]],
            s=75,
            label=plot_label,
        )
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Best development BLEU")
    plt.title("Hyperparameter tuning (marker: batch size 32/64/128)")
    plt.grid(alpha=0.25)
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.savefig(output_dir / "tuning_results.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="97: NMT hyperparameter tuning")
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bleu-train-samples", type=int, default=1000)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-dev-examples", type=int)
    parser.add_argument("--required-bleu", type=float, default=10.0)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    trials = select_trials(args.max_trials, args.seed)
    results = [run_trial(args, *trial) for trial in trials]
    save_results(results, args.output_dir)
    best = max(results, key=lambda row: row["best_dev_bleu"])
    print("\nBest trial")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    if best["best_dev_bleu"] >= args.required_bleu:
        print(f"Confirmed: development BLEU >= {args.required_bleu:.1f}")
    else:
        print(
            f"WARNING: best BLEU {best['best_dev_bleu']:.2f} is below "
            f"{args.required_bleu:.1f}; increase --epochs or --max-trials."
        )


if __name__ == "__main__":
    main()
