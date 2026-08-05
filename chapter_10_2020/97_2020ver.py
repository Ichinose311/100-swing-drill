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


# ===== EXECUTION RESULT =====
# Log: outputs/97_tuning/full_run.log
# Epoch 08 | 06100/13759 | loss 5.3935 | ppl 219.97 | lr 5.928894e-06 | 209.8s
# Epoch 08 | 06200/13759 | loss 5.3933 | ppl 219.93 | lr 5.926002e-06 | 213.2s
# Epoch 08 | 06300/13759 | loss 5.3931 | ppl 219.88 | lr 5.923114e-06 | 216.7s
# Epoch 08 | 06400/13759 | loss 5.3932 | ppl 219.90 | lr 5.92023e-06 | 220.1s
# Epoch 08 | 06500/13759 | loss 5.3930 | ppl 219.86 | lr 5.91735e-06 | 223.4s
# Epoch 08 | 06600/13759 | loss 5.3929 | ppl 219.83 | lr 5.914474e-06 | 226.9s
# Epoch 08 | 06700/13759 | loss 5.3928 | ppl 219.81 | lr 5.911603e-06 | 230.3s
# Epoch 08 | 06800/13759 | loss 5.3925 | ppl 219.75 | lr 5.908735e-06 | 233.7s
# Epoch 08 | 06900/13759 | loss 5.3923 | ppl 219.71 | lr 5.905872e-06 | 237.1s
# Epoch 08 | 07000/13759 | loss 5.3923 | ppl 219.71 | lr 5.903013e-06 | 240.4s
# Epoch 08 | 07100/13759 | loss 5.3923 | ppl 219.70 | lr 5.900159e-06 | 243.9s
# Epoch 08 | 07200/13759 | loss 5.3921 | ppl 219.66 | lr 5.897308e-06 | 247.2s
# Epoch 08 | 07300/13759 | loss 5.3918 | ppl 219.60 | lr 5.894461e-06 | 250.6s
# Epoch 08 | 07400/13759 | loss 5.3919 | ppl 219.63 | lr 5.891619e-06 | 254.1s
# Epoch 08 | 07500/13759 | loss 5.3918 | ppl 219.60 | lr 5.888781e-06 | 257.6s
# Epoch 08 | 07600/13759 | loss 5.3918 | ppl 219.59 | lr 5.885946e-06 | 261.0s
# Epoch 08 | 07700/13759 | loss 5.3915 | ppl 219.53 | lr 5.883116e-06 | 264.6s
# Epoch 08 | 07800/13759 | loss 5.3913 | ppl 219.49 | lr 5.88029e-06 | 268.0s
# Epoch 08 | 07900/13759 | loss 5.3914 | ppl 219.51 | lr 5.877468e-06 | 271.5s
# Epoch 08 | 08000/13759 | loss 5.3912 | ppl 219.47 | lr 5.87465e-06 | 274.9s
# Epoch 08 | 08100/13759 | loss 5.3912 | ppl 219.46 | lr 5.871837e-06 | 278.3s
# Epoch 08 | 08200/13759 | loss 5.3911 | ppl 219.44 | lr 5.869027e-06 | 281.6s
# Epoch 08 | 08300/13759 | loss 5.3911 | ppl 219.45 | lr 5.866221e-06 | 285.0s
# Epoch 08 | 08400/13759 | loss 5.3911 | ppl 219.44 | lr 5.863419e-06 | 288.4s
# Epoch 08 | 08500/13759 | loss 5.3910 | ppl 219.43 | lr 5.860622e-06 | 291.9s
# Epoch 08 | 08600/13759 | loss 5.3909 | ppl 219.39 | lr 5.857828e-06 | 295.4s
# Epoch 08 | 08700/13759 | loss 5.3906 | ppl 219.33 | lr 5.855038e-06 | 298.8s
# Epoch 08 | 08800/13759 | loss 5.3903 | ppl 219.26 | lr 5.852252e-06 | 302.3s
# Epoch 08 | 08900/13759 | loss 5.3902 | ppl 219.24 | lr 5.84947e-06 | 305.8s
# Epoch 08 | 09000/13759 | loss 5.3899 | ppl 219.19 | lr 5.846693e-06 | 309.2s
# Epoch 08 | 09100/13759 | loss 5.3898 | ppl 219.16 | lr 5.843919e-06 | 312.7s
# Epoch 08 | 09200/13759 | loss 5.3897 | ppl 219.14 | lr 5.841149e-06 | 316.2s
# Epoch 08 | 09300/13759 | loss 5.3895 | ppl 219.10 | lr 5.838383e-06 | 319.6s
# Epoch 08 | 09400/13759 | loss 5.3893 | ppl 219.05 | lr 5.835621e-06 | 323.1s
# Epoch 08 | 09500/13759 | loss 5.3891 | ppl 219.00 | lr 5.832862e-06 | 326.5s
# Epoch 08 | 09600/13759 | loss 5.3890 | ppl 218.99 | lr 5.830108e-06 | 329.9s
# Epoch 08 | 09700/13759 | loss 5.3888 | ppl 218.94 | lr 5.827358e-06 | 333.4s
# Epoch 08 | 09800/13759 | loss 5.3886 | ppl 218.89 | lr 5.824611e-06 | 336.8s
# Epoch 08 | 09900/13759 | loss 5.3886 | ppl 218.89 | lr 5.821869e-06 | 340.2s
# Epoch 08 | 10000/13759 | loss 5.3886 | ppl 218.91 | lr 5.81913e-06 | 343.6s
# Epoch 08 | 10100/13759 | loss 5.3884 | ppl 218.86 | lr 5.816395e-06 | 347.1s
# Epoch 08 | 10200/13759 | loss 5.3880 | ppl 218.77 | lr 5.813664e-06 | 350.5s
# Epoch 08 | 10300/13759 | loss 5.3880 | ppl 218.76 | lr 5.810937e-06 | 354.1s
# Epoch 08 | 10400/13759 | loss 5.3877 | ppl 218.69 | lr 5.808214e-06 | 357.5s
# Epoch 08 | 10500/13759 | loss 5.3876 | ppl 218.67 | lr 5.805494e-06 | 360.9s
# Epoch 08 | 10600/13759 | loss 5.3876 | ppl 218.68 | lr 5.802778e-06 | 364.4s
# Epoch 08 | 10700/13759 | loss 5.3877 | ppl 218.70 | lr 5.800067e-06 | 367.8s
# Epoch 08 | 10800/13759 | loss 5.3877 | ppl 218.70 | lr 5.797359e-06 | 371.3s
# Epoch 08 | 10900/13759 | loss 5.3877 | ppl 218.70 | lr 5.794654e-06 | 374.8s
# Epoch 08 | 11000/13759 | loss 5.3876 | ppl 218.67 | lr 5.791954e-06 | 378.3s
# Epoch 08 | 11100/13759 | loss 5.3874 | ppl 218.64 | lr 5.789257e-06 | 381.7s
# Epoch 08 | 11200/13759 | loss 5.3875 | ppl 218.65 | lr 5.786564e-06 | 385.2s
# Epoch 08 | 11300/13759 | loss 5.3876 | ppl 218.68 | lr 5.783875e-06 | 388.6s
# Epoch 08 | 11400/13759 | loss 5.3877 | ppl 218.69 | lr 5.781189e-06 | 392.0s
# Epoch 08 | 11500/13759 | loss 5.3876 | ppl 218.67 | lr 5.778508e-06 | 395.5s
# Epoch 08 | 11600/13759 | loss 5.3874 | ppl 218.64 | lr 5.77583e-06 | 398.9s
# Epoch 08 | 11700/13759 | loss 5.3872 | ppl 218.59 | lr 5.773155e-06 | 402.4s
# Epoch 08 | 11800/13759 | loss 5.3870 | ppl 218.56 | lr 5.770485e-06 | 405.8s
# Epoch 08 | 11900/13759 | loss 5.3868 | ppl 218.51 | lr 5.767818e-06 | 409.1s
# Epoch 08 | 12000/13759 | loss 5.3868 | ppl 218.51 | lr 5.765155e-06 | 412.6s
# Epoch 08 | 12100/13759 | loss 5.3867 | ppl 218.49 | lr 5.762495e-06 | 416.1s
# Epoch 08 | 12200/13759 | loss 5.3867 | ppl 218.47 | lr 5.759839e-06 | 419.5s
# Epoch 08 | 12300/13759 | loss 5.3866 | ppl 218.45 | lr 5.757187e-06 | 423.0s
# Epoch 08 | 12400/13759 | loss 5.3864 | ppl 218.42 | lr 5.754539e-06 | 426.5s
# Epoch 08 | 12500/13759 | loss 5.3864 | ppl 218.41 | lr 5.751894e-06 | 429.9s
# Epoch 08 | 12600/13759 | loss 5.3864 | ppl 218.42 | lr 5.749253e-06 | 433.3s
# Epoch 08 | 12700/13759 | loss 5.3863 | ppl 218.39 | lr 5.746615e-06 | 436.6s
# Epoch 08 | 12800/13759 | loss 5.3862 | ppl 218.36 | lr 5.743981e-06 | 440.1s
# Epoch 08 | 12900/13759 | loss 5.3861 | ppl 218.35 | lr 5.741351e-06 | 443.5s
# Epoch 08 | 13000/13759 | loss 5.3860 | ppl 218.33 | lr 5.738724e-06 | 447.0s
# Epoch 08 | 13100/13759 | loss 5.3860 | ppl 218.33 | lr 5.736101e-06 | 450.5s
# Epoch 08 | 13200/13759 | loss 5.3858 | ppl 218.29 | lr 5.733481e-06 | 453.9s
# Epoch 08 | 13300/13759 | loss 5.3859 | ppl 218.31 | lr 5.730866e-06 | 457.4s
# Epoch 08 | 13400/13759 | loss 5.3857 | ppl 218.27 | lr 5.728253e-06 | 460.7s
# Epoch 08 | 13500/13759 | loss 5.3856 | ppl 218.24 | lr 5.725644e-06 | 464.2s
# Epoch 08 | 13600/13759 | loss 5.3855 | ppl 218.22 | lr 5.723039e-06 | 467.7s
# Epoch 08 | 13700/13759 | loss 5.3854 | ppl 218.19 | lr 5.720438e-06 | 471.1s
# Epoch 08 complete | train loss 5.3852 | dev loss 5.4435 | train BLEU 2.87 | dev BLEU 2.14 | 481.9s
# {
#   "best_epoch": 8,
#   "best_dev_bleu": 2.1423112393893193,
#   "best_dev_loss": 5.443521135842823,
#   "best_model": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/97_tuning/bs32_lr3em05_radam/best_model.pt",
#   "batch_size": 32,
#   "learning_rate": 3e-05,
#   "optimizer": "radam"
# }
# 
# Best trial
# {
#   "trial": "bs64_lr1em03_adamw",
#   "batch_size": 64,
#   "learning_rate": 0.001,
#   "optimizer": "adamw",
#   "best_epoch": 8,
#   "best_dev_bleu": 16.53206190327405,
#   "best_dev_loss": 3.508445771884429,
#   "best_model": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/97_tuning/bs64_lr1em03_adamw/best_model.pt"
# }
# Confirmed: development BLEU >= 10.0
