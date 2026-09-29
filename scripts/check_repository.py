"""Offline source, exercise-coverage and local Markdown link checks."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def sources():
    for directory in [*ROOT.glob("chapter_*"), ROOT / "scripts", ROOT / "tests"]:
        yield from directory.rglob("*.py")


def check():
    files = list(sources())
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        ast.parse(path.read_text(encoding="utf-8"))
    for chapter in range(1, 11):
        for number in range((chapter - 1) * 10, chapter * 10):
            assert (ROOT / f"chapter_{chapter}" / f"{number:02}.py").is_file()
    for number in range(90, 100):
        assert (ROOT / "chapter_10_2020" / f"{number}_2020ver.py").is_file()
    count = 0
    documents = [ROOT / "README.md", *ROOT.glob("docs/*.md"), *ROOT.glob("results/*.md")]
    for path in documents:
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"[a-zA-Z]+://", target):
                continue
            target = unquote(target)
            name, _, fragment = target.partition("#")
            destination = path.parent / name if name else path
            assert destination.exists(), f"Broken link in {path.name}: {target}"
            if fragment:
                body = destination.read_text(encoding="utf-8")
                anchors = set(re.findall(r'<a id="([^"]+)"', body))
                for heading in re.findall(r"^#+\s+(.+)$", body, re.M):
                    anchors.add(re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-"))
                assert fragment in anchors, f"Broken anchor in {path.name}: {target}"
            count += 1
    print(f"PASS: {len(files)} Python sources, 110 exercise entries, {count} local links")


if __name__ == "__main__":
    check()
