from __future__ import annotations

import json
from pathlib import Path

import wandb


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "96_tensorboard"
OUTPUT_DIR = BASE_DIR / "outputs" / "96_tensorboard"


def main() -> None:
    history = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((MODEL_DIR / "summary.json").read_text(encoding="utf-8"))

    run = wandb.init(
        project="100-swing-drill-chapter10",
        name="task96-sentencepiece-transformer",
        config={
            "task": 96,
            "model": "TransformerNMT",
            "tokenizer": "SentencePiece BPE",
            "batch_size": summary["batch_size"],
            "learning_rate": summary["learning_rate"],
            "optimizer": summary["optimizer"],
        },
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("dev/*", step_metric="epoch")
    run.define_metric("learning_rate", step_metric="epoch")

    for row in history:
        run.log(
            {
                "epoch": row["epoch"],
                "train/loss": row["train_loss"],
                "dev/loss": row["dev_loss"],
                "train/BLEU": row["train_bleu"],
                "dev/BLEU": row["dev_bleu"],
                "learning_rate": row["learning_rate"],
            }
        )

    run.summary["best_epoch"] = summary["best_epoch"]
    run.summary["best_dev_bleu"] = summary["best_dev_bleu"]
    run.summary["best_dev_loss"] = summary["best_dev_loss"]
    url = run.url
    run.finish()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "wandb_url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"W&B URL: {url}")


if __name__ == "__main__":
    main()
