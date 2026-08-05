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
