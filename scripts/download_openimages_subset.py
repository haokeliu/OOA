#!/usr/bin/env python3
"""Resumably download an explicit Open Images subset from the public S3 mirror."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LINE_PATTERN = re.compile(r"^(train|validation|test)/([0-9a-fA-F]+)$")
BASE_URL = "https://open-images-dataset.s3.amazonaws.com"


def download_one(split: str, image_id: str, output_root: Path) -> str:
    directory = output_root / split
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{image_id}.jpg"
    if destination.is_file() and destination.stat().st_size > 0:
        return "resumed"
    temporary = destination.with_suffix(".part")
    url = f"{BASE_URL}/{split}/{image_id}.jpg"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                temporary.write_bytes(response.read())
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"empty image response: {url}")
            temporary.replace(destination)
            return "downloaded"
        except Exception as error:  # noqa: BLE001 - retries preserve the final error
            last_error = error
            if temporary.exists():
                temporary.unlink()
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to download {split}/{image_id}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_list", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    entries = []
    for line in args.image_list.read_text().splitlines():
        match = LINE_PATTERN.fullmatch(line.strip())
        if match is None:
            raise SystemExit(f"invalid image-list row: {line!r}")
        entries.append(match.groups())
    if len(entries) != len(set(entries)):
        raise SystemExit("duplicate image-list entries")

    counts = {"downloaded": 0, "resumed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, split, image_id, args.output_root): (split, image_id)
            for split, image_id in entries
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            counts[status] += 1
            if completed % 100 == 0 or completed == len(entries):
                print(
                    json.dumps(
                        {"completed": completed, "total": len(entries), **counts},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
