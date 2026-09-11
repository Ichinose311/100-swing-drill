"""Place verified, separately obtained datasets where the exercises expect them."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(Path(__file__).with_name("datasets.json").read_text())


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare(name: str, source: Path, root: Path = ROOT) -> None:
    spec = MANIFEST[name]
    if digest(source) != spec["sha256"]:
        raise ValueError("SHA-256 mismatch; verify the source/version before updating the manifest")
    targets = [root / p for p in spec["destinations"]]
    # Validate every existing target before writing any destination.
    for target in targets:
        if target.exists() and digest(target) != spec["sha256"]:
            raise FileExistsError(f"Refusing to overwrite a different file: {target.name}")
    if name == "wiki":
        unpacked = root / "chapter_3" / "jawiki-country.json"
        with gzip.open(source, "rb") as stream:
            expected = hashlib.sha256(stream.read()).hexdigest()
        if unpacked.exists() and digest(unpacked) != expected:
            raise FileExistsError("Refusing to overwrite a different jawiki-country.json")
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        print(f"Ready: {target.relative_to(root)}")
    if name == "wiki" and not unpacked.exists():
        with gzip.open(source, "rb") as stream, unpacked.open("wb") as output:
            shutil.copyfileobj(stream, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(MANIFEST))
    parser.add_argument("--from-file", type=Path, help="Verify and place an existing download")
    args = parser.parse_args()
    if args.from_file:
        prepare(args.dataset, args.from_file)
        return
    spec = MANIFEST[args.dataset]
    if not spec["url"]:
        parser.error(
            "Obtain this dataset from its official provider, then use --from-file "
            "(see README.md)"
        )
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "download"
        with requests.get(
            spec["url"],
            headers={"User-Agent": "NLP100Exercises/1.0"},
            stream=True,
            timeout=60,
        ) as response, source.open("wb") as output:
            response.raise_for_status()
            total = 0
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > spec["bytes"]:
                    raise ValueError("Unexpected download size; verify the data source/version")
                output.write(chunk)
        prepare(args.dataset, source)


if __name__ == "__main__":
    main()
