#!/usr/bin/env python3
"""Select the frozen fit-only V3 relation panel."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_SIZE = 30
MIN_CELL_SUPPORT = 200


def main() -> int:
    mapping = json.loads(
        (REPO_ROOT / "data/manifests/category_mapping.json").read_text(encoding="utf-8")
    )
    with (REPO_ROOT / "data/manifests/fit_relation_counts.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {
                key: float(value) if key == "smoothed_lift" else int(value)
                for key, value in raw.items()
            }
            if min(row[key] for key in ("n11", "n10", "n01", "n00")) >= MIN_CELL_SUPPORT:
                rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["smoothed_lift"],
            -row["n11"],
            row["source_id"],
            row["target_id"],
        )
    )
    target_ids: set[int] = set()
    unordered_pairs: set[tuple[int, int]] = set()
    panel = []
    for row in rows:
        unordered = tuple(sorted((row["source_id"], row["target_id"])))
        if row["target_id"] in target_ids or unordered in unordered_pairs:
            continue
        target_ids.add(row["target_id"])
        unordered_pairs.add(unordered)
        panel.append(
            {
                **row,
                "source_name": mapping["names"][row["source_id"]],
                "target_name": mapping["names"][row["target_id"]],
                "pair_id": (
                    f"{mapping['names'][row['source_id']].replace(' ', '_')}_to_"
                    f"{mapping['names'][row['target_id']].replace(' ', '_')}"
                ),
            }
        )
        if len(panel) == PANEL_SIZE:
            break
    if len(panel) != PANEL_SIZE:
        raise RuntimeError(f"only {len(panel)} relations satisfy the frozen panel rule")
    path = REPO_ROOT / "data/manifests/v3_relation_panel.json"
    path.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "selection_data": "fit_relation_counts_only",
        "test_accessed": False,
        "panel_size": PANEL_SIZE,
        "minimum_cell_support": MIN_CELL_SUPPORT,
        "unique_targets": True,
        "unique_unordered_pairs": True,
        "ranking": "smoothed_lift_desc_then_n11_desc_then_ids",
        "relations": panel,
    }
    (REPO_ROOT / "reports/v3_relation_panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
