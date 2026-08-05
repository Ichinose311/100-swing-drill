from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MARKER = "# ===== EXECUTION RESULT ====="

SPECS = {
    "94_2020ver.py": ("outputs/94_beam_search/full_run.log", 160),
    "95_2020ver.py": ("outputs/95_sentencepiece/full_run.log", 45),
    "96_2020ver.py": ("outputs/96_tensorboard/full_run.log", 45),
    "97_2020ver.py": ("outputs/97_tuning/full_run.log", 100),
    "98_2020ver.py": ("outputs/98_domain_adapt/full_run.log", 120),
    "99_2020ver.py": ("outputs/99_server/full_run.log", 100),
}


def append_result(code_name: str, log_name: str, line_count: int) -> None:
    code_path = BASE_DIR / code_name
    log_path = BASE_DIR / log_name
    if not code_path.exists() or not log_path.exists():
        print(f"Skip: {code_name} ({log_name} is missing)")
        return

    code = code_path.read_text(encoding="utf-8")
    if MARKER in code:
        code = code.split(MARKER, maxsplit=1)[0].rstrip()

    log_lines = log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[-line_count:]
    result_lines = [MARKER, f"# Log: {log_name}"]
    result_lines.extend(f"# {line}" for line in log_lines)

    if code_name == "96_2020ver.py":
        url_path = BASE_DIR / "outputs" / "96_tensorboard" / "wandb_url.txt"
        if url_path.exists():
            result_lines.append(f"# W&B URL: {url_path.read_text().strip()}")

    code_path.write_text(
        code + "\n\n" + "\n".join(result_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Updated: {code_path}")


def main() -> None:
    for code_name, (log_name, line_count) in SPECS.items():
        append_result(code_name, log_name, line_count)


if __name__ == "__main__":
    main()
