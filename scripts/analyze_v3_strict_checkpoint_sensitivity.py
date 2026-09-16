#!/usr/bin/env python3
"""Post-reveal sensitivity using only selectors that never saw checkpoint labels."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_v3a_cluster_robustness import cross_validated_prediction

STRICT_REPRESENTATIONS = ("confidence", "summary", "view_logits")
BOOTSTRAP_REPLICATES = 20_000
SIGN_FLIP_REPLICATES = 100_000
RANDOM_SEED = 20260911


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return float(spearmanr(first, second).statistic)


def coco_analysis(rows: list[dict[str, object]], generator: np.random.Generator) -> dict:
    selected_rows = [
        row for row in rows if row["representation"] in STRICT_REPRESENTATIONS
    ]
    target = np.asarray(
        [row["modelval_realized"] for row in selected_rows], dtype=np.float64
    )
    oracle_prediction = cross_validated_prediction(selected_rows, False)
    observable_prediction = cross_validated_prediction(selected_rows, True)
    relations = sorted({str(row["pair_id"]) for row in selected_rows})
    relation_indices = [
        np.asarray(
            [
                index
                for index, row in enumerate(selected_rows)
                if row["pair_id"] == relation
            ]
        )
        for relation in relations
    ]
    oracle_r2 = float(r2_score(target, oracle_prediction))
    observable_r2 = float(r2_score(target, observable_prediction))
    r2_gain = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen_relations = generator.integers(0, len(relations), size=len(relations))
        chosen = np.concatenate([relation_indices[index] for index in chosen_relations])
        r2_gain[replicate] = r2_score(
            target[chosen], observable_prediction[chosen]
        ) - r2_score(target[chosen], oracle_prediction[chosen])
    per_relation_sse_gain = np.asarray(
        [
            np.sum((target[index] - oracle_prediction[index]) ** 2)
            - np.sum((target[index] - observable_prediction[index]) ** 2)
            for index in relation_indices
        ]
    )
    observed_sse_gain = float(per_relation_sse_gain.sum())
    null_sse_gain = (
        generator.choice(
            (-1.0, 1.0), size=(SIGN_FLIP_REPLICATES, len(relations))
        )
        @ per_relation_sse_gain
    )
    return {
        "relation_count": len(relations),
        "task_count": len(selected_rows),
        "oracle_only": {
            "r_squared": oracle_r2,
            "spearman": spearman(target, oracle_prediction),
            "mean_absolute_error": float(np.mean(np.abs(target - oracle_prediction))),
        },
        "oracle_plus_observable": {
            "r_squared": observable_r2,
            "spearman": spearman(target, observable_prediction),
            "mean_absolute_error": float(
                np.mean(np.abs(target - observable_prediction))
            ),
        },
        "r_squared_gain": observable_r2 - oracle_r2,
        "relation_cluster_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "r_squared_gain_95_percentile_interval": np.quantile(
                r2_gain, [0.025, 0.975]
            ).tolist(),
            "fraction_r_squared_gain_positive": float(np.mean(r2_gain > 0)),
        },
        "relation_block_sign_flip": {
            "replicates": SIGN_FLIP_REPLICATES,
            "one_sided_p": float(
                (1 + np.sum(null_sse_gain >= observed_sse_gain))
                / (SIGN_FLIP_REPLICATES + 1)
            ),
        },
    }


def external_dataset(source_path: Path, realized_key: str) -> tuple[dict, list[dict]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = [
        row for row in source["rows"] if row["representation"] in STRICT_REPRESENTATIONS
    ]
    oracle = np.asarray([row["checkpoint_oracle"] for row in rows])
    observable = np.asarray([row["checkpoint_observable"] for row in rows])
    realized = np.asarray([row[realized_key] for row in rows])
    per_seed = {}
    for seed in (17, 29, 43):
        seed_rows = [row for row in rows if row["seed"] == seed]
        per_seed[str(seed)] = {
            "oracle_spearman": spearman(
                np.asarray([row["checkpoint_oracle"] for row in seed_rows]),
                np.asarray([row[realized_key] for row in seed_rows]),
            ),
            "observable_spearman": spearman(
                np.asarray([row["checkpoint_observable"] for row in seed_rows]),
                np.asarray([row[realized_key] for row in seed_rows]),
            ),
        }
    oracle_correlation = spearman(oracle, realized)
    observable_correlation = spearman(observable, realized)
    return (
        {
            "task_count": len(rows),
            "oracle_spearman": oracle_correlation,
            "observable_spearman": observable_correlation,
            "advantage": observable_correlation - oracle_correlation,
            "per_seed": per_seed,
        },
        [
            {
                "pair_id": row["pair_id"],
                "dataset": source["protocol"],
                "seed": row["seed"],
                "representation": row["representation"],
                "oracle": row["checkpoint_oracle"],
                "observable": row["checkpoint_observable"],
                "realized": row[realized_key],
            }
            for row in rows
        ],
    )


def combined_external(
    rows: list[dict[str, object]], generator: np.random.Generator
) -> dict:
    relations = sorted({str(row["pair_id"]) for row in rows})
    tasks = sorted(
        {
            (str(row["dataset"]), int(row["seed"]), str(row["representation"]))
            for row in rows
        }
    )
    lookup = {
        (
            str(row["pair_id"]),
            str(row["dataset"]),
            int(row["seed"]),
            str(row["representation"]),
        ): row
        for row in rows
    }
    arrays = {
        field: np.asarray(
            [
                [lookup[(relation, *task)][field] for task in tasks]
                for relation in relations
            ],
            dtype=np.float64,
        )
        for field in ("oracle", "observable", "realized")
    }
    realized = arrays["realized"].reshape(-1)
    oracle = arrays["oracle"].reshape(-1)
    observable = arrays["observable"].reshape(-1)
    oracle_correlation = spearman(oracle, realized)
    observable_correlation = spearman(observable, realized)
    advantage = observable_correlation - oracle_correlation
    swap_advantages = []
    for bits in itertools.product((False, True), repeat=len(relations)):
        swap = np.asarray(bits)[:, None]
        first = np.where(swap, arrays["observable"], arrays["oracle"])
        second = np.where(swap, arrays["oracle"], arrays["observable"])
        swap_advantages.append(
            spearman(second.reshape(-1), realized)
            - spearman(first.reshape(-1), realized)
        )
    bootstrap_advantage = np.empty(BOOTSTRAP_REPLICATES)
    bootstrap_observable = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, len(relations), size=len(relations))
        selected_oracle = arrays["oracle"][selected].reshape(-1)
        selected_observable = arrays["observable"][selected].reshape(-1)
        selected_realized = arrays["realized"][selected].reshape(-1)
        bootstrap_observable[replicate] = spearman(
            selected_observable, selected_realized
        )
        bootstrap_advantage[replicate] = bootstrap_observable[
            replicate
        ] - spearman(selected_oracle, selected_realized)
    return {
        "relation_count": len(relations),
        "task_count": len(rows),
        "oracle_spearman": oracle_correlation,
        "observable_spearman": observable_correlation,
        "advantage": advantage,
        "relation_cluster_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "observable_spearman_95_percentile_interval": np.quantile(
                bootstrap_observable, [0.025, 0.975]
            ).tolist(),
            "advantage_95_percentile_interval": np.quantile(
                bootstrap_advantage, [0.025, 0.975]
            ).tolist(),
            "fraction_advantage_positive": float(
                np.mean(bootstrap_advantage > 0)
            ),
        },
        "exact_relation_block_predictor_swap": {
            "assignment_count": len(swap_advantages),
            "one_sided_p_advantage": float(
                np.mean(np.asarray(swap_advantages) >= advantage)
            ),
        },
    }


def main() -> int:
    generator = np.random.default_rng(RANDOM_SEED)
    coco_source = json.loads(
        (REPO_ROOT / "reports/v3a_observability_law.json").read_text(
            encoding="utf-8"
        )
    )
    voc2007, rows2007 = external_dataset(
        REPO_ROOT / "reports/v3b_voc_external_confirmation.json", "test_realized"
    )
    voc2012, rows2012 = external_dataset(
        REPO_ROOT / "reports/v3c_voc2012_external_confirmation.json",
        "voc2012_realized",
    )
    report = {
        "schema_version": 1,
        "protocol": "v3-posthoc-strict-checkpoint-sensitivity",
        "analysis_status": (
            "posthoc scientific QA after external label reveal; no model fitting or "
            "prediction rerun"
        ),
        "strict_representations": list(STRICT_REPRESENTATIONS),
        "exclusion_reason": (
            "LogitMLP and RCVI used the nominal checkpoint split for early stopping; "
            "tree hyperparameters were fixed and never saw checkpoint labels"
        ),
        "coco_modelval": coco_analysis(coco_source["rows"], generator),
        "voc2007_test": voc2007,
        "voc2012_val": voc2012,
        "combined_external": combined_external(rows2007 + rows2012, generator),
    }
    output_path = REPO_ROOT / "reports/v3_strict_checkpoint_sensitivity.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
