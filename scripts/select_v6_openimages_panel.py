#!/usr/bin/env python3
"""Select the V6 relation panel from disclosed V5 validation support only."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = REPO_ROOT / "data/manifests/v5_openimages_class_mapping.json"
SUPPORT_PATH = REPO_ROOT / "reports/v5_openimages_preparation.json"
OUTPUT_PATH = REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json"
MINIMUM_VALIDATION_POSITIVE = 20
MINIMUM_VALIDATION_NEGATIVE = 20


def main() -> int:
    mapping = json.loads(MAPPING_PATH.read_text())
    support = json.loads(SUPPORT_PATH.read_text())
    support_by_pair = {row["pair_id"]: row for row in support["relation_support"]}
    selected = []
    for relation in mapping["relation_mapping"]:
        row = support_by_pair[relation["pair_id"]]
        positive = row["adaptation_positive"] + row["evaluation_positive"]
        negative = row["adaptation_negative"] + row["evaluation_negative"]
        if positive < MINIMUM_VALIDATION_POSITIVE or negative < MINIMUM_VALIDATION_NEGATIVE:
            continue
        selected.append(
            {
                **relation,
                "validation_verified_positive": positive,
                "validation_verified_negative": negative,
                "selection_rule": "validation_positive>=20_and_validation_negative>=20",
            }
        )
    OUTPUT_PATH.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_relation_count": len(selected),
                "pair_ids": [row["pair_id"] for row in selected],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
