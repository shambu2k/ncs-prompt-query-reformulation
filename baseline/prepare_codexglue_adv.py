#!/usr/bin/env python3
"""Reconstruct the official CodeXGLUE AdvTest JSONL splits from dataset.zip.

The upstream archive already contains:
  - train.txt / valid.txt / test.txt with URL order
  - test_code.jsonl with the code/docstring payload for valid + test

The official preprocess script only uses the train count to assign idx values
for valid/test, so we can reproduce valid.jsonl and test.jsonl exactly without
downloading the full training corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def read_url_list(zip_file: ZipFile, member_name: str) -> list[str]:
    with zip_file.open(member_name) as handle:
        return [line.decode("utf-8").strip() for line in handle if line.strip()]


def count_lines(zip_file: ZipFile, member_name: str) -> int:
    with zip_file.open(member_name) as handle:
        return sum(1 for _ in handle)


def load_records(
    zip_file: ZipFile,
    member_name: str,
    wanted_urls: set[str],
) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with zip_file.open(member_name) as handle:
        for raw_line in handle:
            item = json.loads(raw_line)
            url = item["url"]
            if url in wanted_urls:
                records[url] = item
    missing = wanted_urls.difference(records)
    if missing:
        sample = next(iter(missing))
        raise KeyError(f"Missing {len(missing)} records from {member_name}; sample URL: {sample}")
    return records


def write_split(
    output_path: Path,
    ordered_urls: list[str],
    records_by_url: dict[str, dict],
    start_idx: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for offset, url in enumerate(ordered_urls):
            record = dict(records_by_url[url])
            record["idx"] = start_idx + offset
            handle.write(json.dumps(record))
            handle.write("\n")
    return start_idx + len(ordered_urls)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-zip",
        default="CodeXGLUE/Text-Code/NL-code-search-Adv/dataset.zip",
        help="Path to the official CodeXGLUE AdvTest dataset.zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline/data",
        help="Directory where valid.jsonl and test.jsonl will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_zip = Path(args.dataset_zip)
    output_dir = Path(args.output_dir)

    with ZipFile(dataset_zip) as zip_file:
        train_count = count_lines(zip_file, "dataset/train.txt")
        valid_urls = read_url_list(zip_file, "dataset/valid.txt")
        test_urls = read_url_list(zip_file, "dataset/test.txt")
        wanted_urls = set(valid_urls) | set(test_urls)
        records_by_url = load_records(zip_file, "dataset/test_code.jsonl", wanted_urls)

    next_idx = write_split(output_dir / "valid.jsonl", valid_urls, records_by_url, train_count)
    final_idx = write_split(output_dir / "test.jsonl", test_urls, records_by_url, next_idx)

    summary = {
        "train_count": train_count,
        "valid_count": len(valid_urls),
        "test_count": len(test_urls),
        "last_assigned_idx": final_idx - 1,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
