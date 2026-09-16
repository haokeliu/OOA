#!/usr/bin/env python3
"""Register the V6 float64-only implementation amendment before rerun."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_EVALUATOR = Path("scripts/evaluate_v6_openimages_test.py")
WRAPPER = Path("scripts/evaluate_v6_openimages_test_float64.py")
FREEZE = Path("reports/v6_openimages_test_freeze.json")
OUTPUT = REPO_ROOT / "reports/v6_openimages_numeric_amendment.json"
EXPECTED_ORIGINAL_HASH = "69aa584fae6a6a1f125f78cdd8f025af1d4b8dc8d1c7291bd17ba1e40b96eed0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit("V6 numerical amendment already exists")
    if (REPO_ROOT / "reports/v6_openimages_test_confirmation.json").exists():
        raise SystemExit("refusing to amend after a V6 performance report exists")
    freeze = json.loads((REPO_ROOT / FREEZE).read_text())
    actual = sha256(REPO_ROOT / ORIGINAL_EVALUATOR)
    frozen_hash = freeze["artifact_sha256"][str(ORIGINAL_EVALUATOR)]
    if actual != EXPECTED_ORIGINAL_HASH or frozen_hash != EXPECTED_ORIGINAL_HASH:
        raise SystemExit("original frozen evaluator hash mismatch")
    report = {
        "schema_version": 1,
        "protocol": "v6-openimages-numerical-precision-amendment",
        "created_utc": datetime.now(UTC).isoformat(),
        "registered_before_performance_report": True,
        "performance_report_existed": False,
        "scientific_protocol_changed": False,
        "relations_models_actions_and_primary_statistic_changed": False,
        "original_run_status": "aborted_before_report_write",
        "observed_diagnostic_only": {
            "maximum_float32_soft_excess_over_endpoint_oracle": 2.363567830343527e-7,
            "frozen_tolerance": 1e-8,
        },
        "amendment": (
            "Promote analytic action and BCE calculations to float64, retaining the frozen "
            "1e-8 oracle-bound tolerance and all scientific choices."
        ),
        "artifact_sha256": {
            str(ORIGINAL_EVALUATOR): actual,
            str(WRAPPER): sha256(REPO_ROOT / WRAPPER),
            str(FREEZE): sha256(REPO_ROOT / FREEZE),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
