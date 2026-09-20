"""Pooled multi-seed target/decoy q-value calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QValueCurve:
    threshold: np.ndarray
    target_count: np.ndarray
    decoy_count_by_seed: np.ndarray
    raw_dt_by_seed: np.ndarray
    plus_one_by_seed: np.ndarray
    raw_dt_mean: np.ndarray
    plus_one_mean: np.ndarray
    plus_one_min: np.ndarray
    plus_one_max: np.ndarray
    qvalue: np.ndarray


def _scores(values: np.ndarray | list[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite values")
    return result


def pooled_multiseed_curve(
    target_scores: np.ndarray | list[float],
    decoy_scores_by_seed: list[np.ndarray | list[float]],
) -> QValueCurve:
    """Count pooled target/decoy PSMs at grouped score ties and compute tail-min q-values.

    Inputs contain only non-empty predictions and all scores are already oriented so
    larger is better. The target set is shared across seeds.
    """
    target = _scores(target_scores, "target_scores")
    decoys = [_scores(values, f"decoy_scores_by_seed[{index}]") for index, values in enumerate(decoy_scores_by_seed)]
    if not decoys:
        raise ValueError("at least one decoy seed is required")
    nonempty = [values for values in [target, *decoys] if values.size]
    if not nonempty:
        empty = np.empty(0, dtype=np.float64)
        empty_counts = np.empty((0, len(decoys)), dtype=np.int64)
        empty_rates = np.empty((0, len(decoys)), dtype=np.float64)
        return QValueCurve(
            empty,
            np.empty(0, dtype=np.int64),
            empty_counts,
            empty_rates,
            empty_rates.copy(),
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    thresholds = np.unique(np.concatenate(nonempty))[::-1]
    target_sorted = np.sort(target)
    target_count = target.size - np.searchsorted(target_sorted, thresholds, side="left")
    decoy_count = np.empty((thresholds.size, len(decoys)), dtype=np.int64)
    for seed_index, values in enumerate(decoys):
        sorted_values = np.sort(values)
        decoy_count[:, seed_index] = values.size - np.searchsorted(sorted_values, thresholds, side="left")

    denominator = np.maximum(target_count, 1)[:, None]
    raw_dt = decoy_count / denominator
    plus_one = (1 + decoy_count) / denominator
    raw_mean = raw_dt.mean(axis=1)
    plus_mean = plus_one.mean(axis=1)
    plus_min = plus_one.min(axis=1)
    plus_max = plus_one.max(axis=1)
    qvalue = np.minimum.accumulate(plus_mean[::-1])[::-1]
    qvalue = np.minimum(qvalue, 1.0)
    return QValueCurve(
        thresholds,
        target_count.astype(np.int64),
        decoy_count,
        raw_dt,
        plus_one,
        raw_mean,
        plus_mean,
        plus_min,
        plus_max,
        qvalue,
    )
