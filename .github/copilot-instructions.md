# AI Coding Guidelines for 100-swing-drill

## Project Overview
This is a Python implementation of "言語処理100本ノック" (Language Processing 100 Drills), a collection of exercises for natural language processing. Each chapter contains numbered Python files (00.py, 01.py, etc.) implementing specific NLP tasks.

## Architecture
- **Structure**: Root contains `main.py` (basic entry point) and `pyproject.toml`. Each chapter is in a separate directory (e.g., `capter_1/`) with numbered exercise files.
- **Independence**: Each `.py` file is a standalone script solving one exercise. No cross-file dependencies.
- **Data Flow**: Exercises typically process hardcoded strings or simple inputs, outputting results to stdout.

## Development Workflow
- **Execution**: Run individual exercises with `uv run python <path/to/file.py>` (e.g., `uv run python capter_1/02.py`)
- **Environment**: Managed by `uv` (Python package manager). Dependencies listed in `pyproject.toml`.
- **Testing**: No formal test suite; validate by running scripts and checking output against expected results.

## Key Patterns
- **String Manipulation**: Heavy use of Python string slicing and operations (e.g., `x[::2]` in `capter_1/01.py` for even-indexed characters, `x[::-1]` in `capter_1/02.py` for reversal).
- **Loop Constructs**: Simple `for` loops for character-by-character processing (e.g., `capter_1/00.py` interleaving strings).
- **Output**: Use `print()` with `end=''` for inline output; avoid unnecessary formatting.
- **Encoding**: Handle Japanese text directly (e.g., UTF-8 strings in `capter_1/00.py`).

## Dependencies
- **numpy**: Imported for potential numerical computations in later chapters.
- **No external APIs**: All exercises use built-in Python or numpy; no web requests or file I/O beyond stdout.

## Conventions
- **File Naming**: `XX.py` where XX is zero-padded exercise number (00-99).
- **Code Style**: Minimal, functional code; no classes or functions unless necessary.
- **Comments**: Sparse; rely on code clarity for NLP concepts.

Reference: [capter_1/00.py](capter_1/00.py) for basic string interleaving, [capter_1/02.py](capter_1/02.py) for reversal pattern.