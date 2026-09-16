#!/usr/bin/env python3
"""Numerical-precision wrapper for the frozen V6 evaluator.

The frozen evaluator and scientific protocol remain unchanged.  This wrapper
only promotes base logits, residuals, labels, probabilities, and actions to
float64 before the existing analytic and BCE calculations.
"""

from __future__ import annotations

import evaluate_v6_openimages_test as frozen
import numpy as np

_bce_gain = frozen.bce_gain
_expected_endpoint_gain = frozen.expected_endpoint_gain
_soft_action = frozen.soft_action


def as_float64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def bce_gain_float64(base, residual, labels, q) -> np.ndarray:
    return _bce_gain(
        as_float64(base),
        as_float64(residual),
        as_float64(labels),
        as_float64(q),
    )


def expected_endpoint_gain_float64(probability, base, residual) -> np.ndarray:
    return _expected_endpoint_gain(
        as_float64(probability), as_float64(base), as_float64(residual)
    )


def soft_action_float64(probability, base, residual) -> np.ndarray:
    return _soft_action(
        as_float64(probability), as_float64(base), as_float64(residual)
    )


def main() -> int:
    frozen.bce_gain = bce_gain_float64
    frozen.expected_endpoint_gain = expected_endpoint_gain_float64
    frozen.soft_action = soft_action_float64
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
