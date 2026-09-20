"""Symmetric peptide-composition collapse and conservative multi-seed competition."""

from __future__ import annotations

from collections.abc import Iterable


def collapse_best(records: Iterable[dict]) -> dict[str, dict]:
    """Keep one highest-scoring PSM per non-empty composition key.

    Exact within-origin score ties are resolved only for row provenance by the
    lowest spectrum index; the composition-level score is unchanged.
    """
    collapsed: dict[str, dict] = {}
    tie_counts: dict[str, int] = {}
    for source in records:
        key = source.get("composition_key")
        if not key:
            continue
        record = dict(source)
        record["score"] = float(record["score"])
        record["spectrum_index"] = int(record["spectrum_index"])
        incumbent = collapsed.get(str(key))
        if incumbent is None or record["score"] > incumbent["score"]:
            collapsed[str(key)] = record
            tie_counts[str(key)] = 1
        elif record["score"] == incumbent["score"]:
            tie_counts[str(key)] += 1
            if record["spectrum_index"] < incumbent["spectrum_index"]:
                collapsed[str(key)] = record
    for key, record in collapsed.items():
        record["within_origin_best_score_tie_count"] = tie_counts[key]
    return collapsed


def conservative_multiseed_competition(
    target_records: Iterable[dict],
    decoy_records_by_seed: dict[str, Iterable[dict]],
) -> dict:
    """Create one target set that is valid against every frozen decoy seed.

    A target composition wins only when its score is strictly greater than the
    best same-key decoy score across every seed. Exact cross-origin ties are
    assigned to the decoy. Each seed otherwise retains its own winning decoys
    for the prespecified per-seed plus-one numerator.
    """
    if not decoy_records_by_seed:
        raise ValueError("at least one decoy seed is required")
    target = collapse_best(target_records)
    decoys = {str(seed): collapse_best(records) for seed, records in decoy_records_by_seed.items()}
    keys = sorted(set(target).union(*(set(values) for values in decoys.values())))
    target_winners = []
    decoy_winners = {seed: [] for seed in decoys}
    competition_rows = []
    for key in keys:
        target_record = target.get(key)
        present_decoys = [values[key] for values in decoys.values() if key in values]
        maximum_decoy_score = max((record["score"] for record in present_decoys), default=None)
        target_wins = target_record is not None and (
            maximum_decoy_score is None or target_record["score"] > maximum_decoy_score
        )
        if target_record is not None:
            target_row = {
                **target_record,
                "composition_key": key,
                "origin": "target",
                "seed": None,
                "maximum_same_key_decoy_score": maximum_decoy_score,
                "competition_winner": target_wins,
                "competition_reason": "strictly_above_every_seed" if target_wins else "lost_or_tied_any_seed",
            }
            competition_rows.append(target_row)
            if target_wins:
                target_winners.append(target_row)
        for seed, values in decoys.items():
            decoy_record = values.get(key)
            if decoy_record is None:
                continue
            decoy_wins = target_record is None or decoy_record["score"] >= target_record["score"]
            decoy_row = {
                **decoy_record,
                "composition_key": key,
                "origin": "decoy",
                "seed": seed,
                "same_key_target_score": None if target_record is None else target_record["score"],
                "competition_winner": decoy_wins,
                "competition_reason": "target_absent_or_decoy_ge_target" if decoy_wins else "below_target",
            }
            competition_rows.append(decoy_row)
            if decoy_wins:
                decoy_winners[seed].append(decoy_row)
    return {
        "target_collapsed": target,
        "decoy_collapsed_by_seed": decoys,
        "target_winners": target_winners,
        "decoy_winners_by_seed": decoy_winners,
        "competition_rows": competition_rows,
    }
