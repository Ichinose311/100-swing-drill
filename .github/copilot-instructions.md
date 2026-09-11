# Development notes

This repository contains independent Python exercise scripts in chapter_1 through
chapter_10 (NLP100 2025), plus chapter_10_2020 (machine translation).
There is no main.py entry point. Follow README.md for chapter-specific setup,
data preparation, and execution. Keep the repository name 100-swing-drill.

- Preserve the instructional purpose and numbered exercise filenames.
- Use paths relative to the script or documented data directory.
- Read Gemini credentials from GEMINI_API_KEY; never embed credential values.
- Do not commit downloaded datasets, model weights, raw logs, caches, or backups.
- Do not paste terminal prompts, personal paths, or account URLs into source files.
- Keep sources and third-party notices up to date when adding data or examples.
- API calls, training, dataset downloads, and W&B uploads require an explicit
  execution request. Use static checks and local fixtures for routine validation.
- Keep reusable results such as anonymized metrics and plots when useful.
