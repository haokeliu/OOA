"""Strict configuration loading for EDCR experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "project": str,
    "protocol_version": str,
    "data": {
        "dataset": str,
        "root": str,
        "split_seed": int,
        "splits": list,
        "test_split": str,
        "pair_count": int,
        "sources_per_target": int,
        "max_overlap_ratio": (int, float),
        "max_single_edit_area_ratio": (int, float),
        "max_combined_edit_area_ratio": (int, float),
    },
    "features": {
        "backbone": str,
        "checkpoint": str,
        "checkpoint_sha256": str,
        "frozen": bool,
        "views": str,
        "crop_ratio": (int, float),
        "pooling_temperature": (int, float),
    },
    "training": {
        "seeds": list,
        "optimizer": str,
        "heads_lr": (int, float),
        "gate_lr": (int, float),
        "weight_decay": (int, float),
        "head_epochs": int,
        "gate_epochs": int,
        "patience": int,
        "cached_batch_size": int,
        "gate_hidden": int,
        "residual_bound": (int, float),
        "gate_mode": str,
        "negative_risk_ratio": (int, float),
        "weak_risk_slack": (int, float),
        "dual_lr": (int, float),
        "weak_delta_epsilon": (int, float),
        "original_weight": (int, float),
    },
    "evaluation": {
        "calibration_recall": (int, float),
        "min_calibration_positives": int,
        "weak_small_area_pixels": (int, float),
        "weak_max_area_ratio": (int, float),
        "bootstrap_repeats": int,
    },
    "execution": {
        "phase": str,
        "allow_paid_services": bool,
        "device": str,
        "feature_microbatch": (str, int),
    },
}


def _validate(value: Any, schema: Any, path: str = "config") -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be an object")
        unknown = sorted(set(value) - set(schema))
        missing = sorted(set(schema) - set(value))
        if unknown:
            raise ValueError(f"unknown keys at {path}: {unknown}")
        if missing:
            raise ValueError(f"missing keys at {path}: {missing}")
        for key, child_schema in schema.items():
            _validate(value[key], child_schema, f"{path}.{key}")
        return
    if not isinstance(value, schema):
        expected = getattr(schema, "__name__", str(schema))
        raise TypeError(f"{path} must be {expected}, got {type(value).__name__}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML and reject unknown or missing keys."""
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must remain JSON-compatible YAML for bootstrap auditing"
        ) from exc
    _validate(config, SCHEMA)
    if abs(sum(config["data"]["splits"]) - 1.0) > 1e-12:
        raise ValueError("config.data.splits must sum to 1")
    if config["features"]["frozen"] is not True:
        raise ValueError("pilot encoder must be frozen")
    if config["execution"]["allow_paid_services"] is not False:
        raise ValueError("pilot forbids paid services")
    return config
