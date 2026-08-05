from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TASK95_PATH = BASE_DIR / "95_2020ver.py"


def load_task95():
    spec = importlib.util.spec_from_file_location("task95", TASK95_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {TASK95_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    task95 = load_task95()
    defaults = [
        "train",
        "--tensorboard",
        "--model-dir",
        str(BASE_DIR / "models" / "96_tensorboard"),
        "--log-dir",
        str(BASE_DIR / "outputs" / "96_tensorboard"),
        "--run-name",
        "kftt_sentencepiece",
    ]
    # Arguments written after these defaults can override them.
    task95.main([*defaults, *sys.argv[1:]])


if __name__ == "__main__":
    main()


# ===== EXECUTION RESULT =====
# Log: outputs/96_tensorboard/full_run.log
# Epoch 10 | 03400/06880 | loss 3.4962 | ppl 32.99 | lr 7.423832e-05 | 117.0s
# Epoch 10 | 03500/06880 | loss 3.4967 | ppl 33.01 | lr 7.418156e-05 | 120.5s
# Epoch 10 | 03600/06880 | loss 3.4970 | ppl 33.02 | lr 7.412493e-05 | 124.0s
# Epoch 10 | 03700/06880 | loss 3.4973 | ppl 33.03 | lr 7.406843e-05 | 127.3s
# Epoch 10 | 03800/06880 | loss 3.4977 | ppl 33.04 | lr 7.401206e-05 | 130.5s
# Epoch 10 | 03900/06880 | loss 3.4977 | ppl 33.04 | lr 7.395581e-05 | 133.7s
# Epoch 10 | 04000/06880 | loss 3.4977 | ppl 33.04 | lr 7.38997e-05 | 136.9s
# Epoch 10 | 04100/06880 | loss 3.4981 | ppl 33.05 | lr 7.384371e-05 | 140.3s
# Epoch 10 | 04200/06880 | loss 3.4981 | ppl 33.05 | lr 7.378785e-05 | 143.7s
# Epoch 10 | 04300/06880 | loss 3.4985 | ppl 33.07 | lr 7.373211e-05 | 147.1s
# Epoch 10 | 04400/06880 | loss 3.4989 | ppl 33.08 | lr 7.36765e-05 | 150.6s
# Epoch 10 | 04500/06880 | loss 3.4993 | ppl 33.09 | lr 7.362102e-05 | 154.1s
# Epoch 10 | 04600/06880 | loss 3.4995 | ppl 33.10 | lr 7.356566e-05 | 157.6s
# Epoch 10 | 04700/06880 | loss 3.4997 | ppl 33.11 | lr 7.351043e-05 | 160.9s
# Epoch 10 | 04800/06880 | loss 3.4997 | ppl 33.11 | lr 7.345532e-05 | 164.2s
# Epoch 10 | 04900/06880 | loss 3.5002 | ppl 33.12 | lr 7.340033e-05 | 167.4s
# Epoch 10 | 05000/06880 | loss 3.5003 | ppl 33.13 | lr 7.334547e-05 | 170.7s
# Epoch 10 | 05100/06880 | loss 3.5010 | ppl 33.15 | lr 7.329073e-05 | 174.1s
# Epoch 10 | 05200/06880 | loss 3.5008 | ppl 33.14 | lr 7.323611e-05 | 177.3s
# Epoch 10 | 05300/06880 | loss 3.5009 | ppl 33.15 | lr 7.318162e-05 | 180.6s
# Epoch 10 | 05400/06880 | loss 3.5011 | ppl 33.15 | lr 7.312724e-05 | 183.9s
# Epoch 10 | 05500/06880 | loss 3.5015 | ppl 33.17 | lr 7.307299e-05 | 187.3s
# Epoch 10 | 05600/06880 | loss 3.5015 | ppl 33.17 | lr 7.301886e-05 | 190.8s
# Epoch 10 | 05700/06880 | loss 3.5015 | ppl 33.16 | lr 7.296485e-05 | 194.3s
# Epoch 10 | 05800/06880 | loss 3.5014 | ppl 33.16 | lr 7.291095e-05 | 197.7s
# Epoch 10 | 05900/06880 | loss 3.5013 | ppl 33.16 | lr 7.285718e-05 | 201.0s
# Epoch 10 | 06000/06880 | loss 3.5014 | ppl 33.16 | lr 7.280353e-05 | 204.3s
# Epoch 10 | 06100/06880 | loss 3.5017 | ppl 33.17 | lr 7.274999e-05 | 207.8s
# Epoch 10 | 06200/06880 | loss 3.5017 | ppl 33.17 | lr 7.269657e-05 | 211.1s
# Epoch 10 | 06300/06880 | loss 3.5017 | ppl 33.17 | lr 7.264327e-05 | 214.5s
# Epoch 10 | 06400/06880 | loss 3.5019 | ppl 33.18 | lr 7.259009e-05 | 217.9s
# Epoch 10 | 06500/06880 | loss 3.5022 | ppl 33.19 | lr 7.253702e-05 | 221.2s
# Epoch 10 | 06600/06880 | loss 3.5022 | ppl 33.19 | lr 7.248407e-05 | 224.7s
# Epoch 10 | 06700/06880 | loss 3.5025 | ppl 33.20 | lr 7.243124e-05 | 228.0s
# Epoch 10 | 06800/06880 | loss 3.5027 | ppl 33.20 | lr 7.237852e-05 | 231.4s
# Epoch 10 complete | train loss 3.5026 | dev loss 3.5672 | train BLEU 20.27 | dev BLEU 16.13 | 243.4s
# {
#   "best_epoch": 10,
#   "best_dev_bleu": 16.132226569783963,
#   "best_dev_loss": 3.5671728337093986,
#   "best_model": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/96_tensorboard/best_model.pt",
#   "batch_size": 64,
#   "learning_rate": 0.0003,
#   "optimizer": "adamw"
# }
