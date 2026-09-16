"""Numerically stable scalar reference functions for gate tests and diagnostics."""

from __future__ import annotations

import math


def bce_with_logits(logit: float, target: int) -> float:
    if target not in (0, 1):
        raise ValueError("target must be binary")
    # max(x, 0) - x*y + log(1 + exp(-abs(x)))
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def loss_difference(base: float, residual: float, target: int) -> float:
    return bce_with_logits(base, target) - bce_with_logits(base + residual, target)


def oracle_gate(base: float, residual: float, target: int) -> int:
    return int(loss_difference(base, residual, target) > 0.0)


def hard_gate_regret(base: float, residual: float, target: int, gate: int) -> float:
    if gate not in (0, 1):
        raise ValueError("gate must be binary")
    chosen = bce_with_logits(base + gate * residual, target)
    oracle = min(bce_with_logits(base, target), bce_with_logits(base + residual, target))
    return chosen - oracle


def update_dual(value: float, violation: float, learning_rate: float) -> float:
    return max(0.0, value + learning_rate * violation)
