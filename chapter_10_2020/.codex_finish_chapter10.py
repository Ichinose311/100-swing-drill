from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time


BASE_DIR = Path(__file__).resolve().parent
PYTHON = Path("/data/student/i2311021/conda_envs/100-swing-drill-gpu/bin/python")


def wait_for_process(pid: int) -> None:
    process_dir = Path("/proc") / str(pid)
    while process_dir.exists():
        print(f"Waiting for training process {pid}...", flush=True)
        time.sleep(20)


def run_logged(command: list[str], log_path: Path, mode: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Running: {' '.join(command)}", flush=True)
    with log_path.open(mode, encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=BASE_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, required=True)
    args = parser.parse_args()

    wait_for_process(args.training_pid)
    print("Training process finished; starting task 98 evaluation.", flush=True)

    run_logged(
        [
            str(PYTHON),
            "-u",
            "98_2020ver.py",
            "evaluate",
            "--baseline-checkpoint",
            "models/97_tuning/bs64_lr1em03_adamw/best_model.pt",
            "--beam-size",
            "5",
        ],
        BASE_DIR / "outputs/98_domain_adapt/full_run.log",
        "a",
    )

    print("Task 98 evaluation finished; starting task 99 smoke test.", flush=True)
    run_logged(
        [
            str(PYTHON),
            "-u",
            "99_2020ver.py",
            "--checkpoint",
            "models/98_domain_adapt/best_model.pt",
            "--beam-size",
            "5",
            "--smoke-text",
            "京都は日本の都市です。",
        ],
        BASE_DIR / "outputs/99_server/full_run.log",
        "w",
    )
    print("Task 99 smoke test finished.", flush=True)


if __name__ == "__main__":
    main()
