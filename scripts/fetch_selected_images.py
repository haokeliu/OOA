#!/usr/bin/env python3
"""Download only images selected for a leakage-safe intervention queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def fetch_one(url: str, destination: Path) -> tuple[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        validate_image(destination)
        return "existing", sha256(destination)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        validate_image(temporary)
        digest = sha256(temporary)
        os.replace(temporary, destination)
        return "downloaded", digest
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="data/interventions/gate/candidates.jsonl")
    parser.add_argument("--data-root", default="data/coco2017")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://images.cocodataset.org/train2017")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    queue_path = (REPO_ROOT / args.queue).resolve()
    data_root = (REPO_ROOT / args.data_root).resolve()
    rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    relative_paths = sorted({row["image_path"] for row in rows})
    if args.dry_run:
        print(
            json.dumps(
                {"unique_images": len(relative_paths), "data_root": str(data_root)}, indent=2
            )
        )
        return 0

    jobs = {
        relative: (
            f"{args.base_url.rstrip('/')}/{Path(relative).name}",
            data_root / relative,
        )
        for relative in relative_paths
    }
    hashes: dict[str, str] = {}
    statuses: dict[str, int] = {"downloaded": 0, "existing": 0, "failed": 0}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_one, url, destination): relative
            for relative, (url, destination) in jobs.items()
        }
        for future in as_completed(futures):
            relative = futures[future]
            try:
                status, digest = future.result()
                statuses[status] += 1
                hashes[relative] = digest
            except Exception as exc:  # noqa: BLE001 - every failed file must be recorded
                statuses["failed"] += 1
                failures[relative] = f"{type(exc).__name__}: {exc}"

    for manifest_name in ("candidates.jsonl", "review_queue.jsonl"):
        manifest_path = queue_path.parent / manifest_name
        manifest_rows = [
            json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                digest = hashes.get(row["image_path"])
                if digest:
                    row["parent_sha256"] = digest
                    row["image_sha256"] = digest
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema_version": 1,
        "queue": str(queue_path.relative_to(REPO_ROOT)),
        "unique_images": len(relative_paths),
        "statuses": statuses,
        "failures": failures,
    }
    report_path = queue_path.parent / "download_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
