"""Import exercise modules without running experiments or downloading models."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE_CHAPTERS = {"chapter_1", "chapter_2", "chapter_3", "chapter_4", "chapter_7"}
CABOCHA_EXERCISES = {"33.py", "34.py", "35.py"}


def load(path):
    name = "exercise_" + path.parent.name + "_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-extras", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    def no_network(*args, **kwargs):
        raise RuntimeError("Network access during import is not allowed")

    socket.socket.connect = no_network
    checked = 0
    skipped = []
    for directory in sorted(ROOT.glob("chapter_*")):
        if not args.all_extras and directory.name not in BASE_CHAPTERS:
            continue
        for path in sorted(directory.glob("*.py")):
            if directory.name == "chapter_4" and path.name in CABOCHA_EXERCISES:
                if importlib.util.find_spec("CaboCha") is None:
                    skipped.append(path.relative_to(ROOT).as_posix())
                    continue
            load(path)
            checked += 1
    load(ROOT / "scripts" / "prepare_data.py")
    print(f"PASS: {checked + 1} modules imported with network blocked")
    if skipped:
        print("SKIP (CaboCha not installed): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
