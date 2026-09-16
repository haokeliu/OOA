import csv
import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_v5_openimages import read_verified_targets
from prepare_v5_openimages import partition


def test_v5_partition_matches_frozen_big_endian_rule() -> None:
    for image_id in ("0000000000000000", "ffffffffffffffff", "abc123"):
        digest = hashlib.sha256(f"v5-oi:{image_id}".encode()).digest()
        expected = (
            "adaptation"
            if int.from_bytes(digest[:8], "big") % 2 == 0
            else "evaluation"
        )
        assert partition(image_id) == expected


def test_read_verified_targets_keeps_only_explicit_requested_labels(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("ImageID", "Source", "LabelName", "Confidence")
        )
        writer.writeheader()
        writer.writerows(
            [
                {"ImageID": "a", "Source": "verification", "LabelName": "/m/x", "Confidence": 1},
                {"ImageID": "b", "Source": "verification", "LabelName": "/m/x", "Confidence": 0},
                {"ImageID": "c", "Source": "verification", "LabelName": "/m/y", "Confidence": 1},
            ]
        )
    assert read_verified_targets(path, {"/m/x"}) == {
        ("a", "/m/x"): 1,
        ("b", "/m/x"): 0,
    }


def test_read_verified_targets_rejects_conflicting_duplicates(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "ImageID,Source,LabelName,Confidence\n"
        "a,verification,/m/x,1\n"
        "a,verification,/m/x,0\n"
    )
    with pytest.raises(RuntimeError, match="conflicting duplicate"):
        read_verified_targets(path, {"/m/x"})
