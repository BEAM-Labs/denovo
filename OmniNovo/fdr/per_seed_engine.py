"""Single-seed target/decoy q-value and composition-competition helpers."""

from __future__ import annotations

import numpy as np

from composition_competition import conservative_multiseed_competition
from qvalue_engine import QValueCurve, pooled_multiseed_curve


ESTIMATORS = ("plus_one", "raw_dt")


def selected_estimate(curve: QValueCurve, estimator: str) -> np.ndarray:
    """Return one seed's cumulative estimate at every grouped score threshold."""
    if estimator == "plus_one":
        return curve.plus_one_by_seed[:, 0]
    if estimator == "raw_dt":
        return curve.raw_dt_by_seed[:, 0]
    raise ValueError(f"unknown estimator: {estimator}")


def selected_qvalues(curve: QValueCurve, estimator: str) -> np.ndarray:
    """Tail-min transform the selected single-seed cumulative estimator."""
    estimates = selected_estimate(curve, estimator)
    if estimates.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.minimum(np.minimum.accumulate(estimates[::-1])[::-1], 1.0)


def map_target_qvalues(target_scores, curve: QValueCurve, estimator: str) -> np.ndarray:
    """Map grouped-tie q-values to target scores without splitting ties."""
    target = np.asarray(target_scores, dtype=np.float64)
    qvalues = selected_qvalues(curve, estimator)
    if curve.threshold.size == 0:
        if target.size:
            raise ValueError("cannot map target scores onto an empty curve")
        return np.empty(0, dtype=np.float64)
    mapping = {float(score): float(qvalue) for score, qvalue in zip(curve.threshold, qvalues)}
    try:
        return np.asarray([mapping[float(score)] for score in target], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"target score is absent from curve thresholds: {error.args[0]}") from error


def single_seed_curve(target_scores, decoy_scores) -> QValueCurve:
    """Build a curve with exactly one decoy seed."""
    return pooled_multiseed_curve(target_scores, [decoy_scores])


def single_seed_composition_competition(target_records, decoy_records, seed: str) -> dict:
    """Run symmetric key competition using only the named seed."""
    result = conservative_multiseed_competition(target_records, {str(seed): decoy_records})
    if set(result["decoy_winners_by_seed"]) != {str(seed)}:
        raise RuntimeError("single-seed composition competition leaked another seed")
    return result

