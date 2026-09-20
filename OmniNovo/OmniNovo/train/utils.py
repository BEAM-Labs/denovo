"""Utility functions for mass calculation, CTC post-processing, and PTM analysis."""
import re
import random
import collections
import itertools
import torch
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict


class PeptideMass_PTM:
    """Calculator for peptide masses including post-translational modifications."""

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        """Initialize the PeptideMass object"""
        self.masses = residues

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z(])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass


def swap_random_neighbor_list(data: List, max_trials: int = 20) -> List:
    """
    Randomly swap adjacent elements (last element stays fixed, and if the first
    element is '(ac)' it also stays fixed), ensuring the result differs from the
    original. Returns a copy of the original if no valid swap is found.
    """
    data = re.split(r"(?<=.)(?=[A-Z])", data)
    n = len(data)
    if n < 3:
        return data.copy()

    if data[0] == "(ac)":
        has_swappable = any(data[i] != data[i+1] for i in range(1, n-2))
        if not has_swappable:
            return data.copy()
        for _ in range(max_trials):
            new_data = data.copy()
            i = random.randint(1, n - 3)
            new_data[i], new_data[i+1] = new_data[i+1], new_data[i]
            if new_data != data:
                return new_data
    else:
        has_swappable = any(data[i] != data[i+1] for i in range(n-2))
        if not has_swappable:
            return data.copy()
        for _ in range(max_trials):
            new_data = data.copy()
            i = random.randint(0, n - 3)
            new_data[i], new_data[i+1] = new_data[i+1], new_data[i]
            if new_data != data:
                return new_data

    return data.copy()


def shuffle_list(data: List, max_trials: int = 20) -> List:
    """
    Randomly shuffle a list (last element stays fixed, and if the first element
    is '(ac)' it also stays fixed), ensuring the result differs from the original.
    """
    data = re.split(r"(?<=.)(?=[A-Z])", data)
    n = len(data)
    if n < 3:
        return data.copy()

    if data[0] == "(ac)":
        prefix = data[1:-1]
        last = data[-1]
        if all(x == prefix[0] for x in prefix):
            return data.copy()
        for _ in range(max_trials):
            shuffled_prefix = random.sample(prefix, len(prefix))
            if shuffled_prefix != prefix:
                return [data[0]] + shuffled_prefix + [last]
    else:
        prefix = data[:-1]
        last = data[-1]
        if all(x == prefix[0] for x in prefix):
            return data.copy()
        for _ in range(max_trials):
            shuffled_prefix = random.sample(prefix, len(prefix))
            if shuffled_prefix != prefix:
                return shuffled_prefix + [last]

    return data.copy()


# PTM modification rules: which amino acids each PTM can modify
PTM_RULES: Dict[str, List[str]] = {
    "(ph)": ["S", "T", "Y"],
    "(me)": ["K", "R", "S"],
    "(di)": ["K", "R"],
    "(tr)": ["K"],
    "(ac)": ["K", "R"],
    "(ox)": ["M"],
    "(ub)": ["K"],
    "(cam)": ["C"],
    "(deam)": ["N", "Q"]
}

def find_ptm_variants(sequence: List[str], rules: Dict[str, List[str]] = PTM_RULES) -> List[List[str]]:
    """
    Find all PTM placement variants for a given peptide sequence.

    Given a peptide with PTMs, generates all valid alternative placements
    of those PTMs on other compatible amino acid sites.

    Args:
        sequence: Peptide sequence as a string or list of tokens.
        rules: Dictionary mapping PTM types to lists of amino acids they can modify.

    Returns:
        List of alternative sequences (as strings), excluding the original.
    """
    if not isinstance(sequence, list):
        sequence = re.split(r"(?<=.)(?=[A-Z(])", sequence)

    original_sequence = sequence.copy()

    # Handle special N-terminal (ac) modification
    has_n_terminal_ac = sequence[0] == "(ac)"
    work_sequence = sequence[1:] if has_n_terminal_ac else sequence

    # Separate amino acids and PTMs
    bare_peptide = [item for item in work_sequence if not item.startswith('(')]
    ptms_in_sequence = [item for item in work_sequence if item.startswith('(')]

    if not ptms_in_sequence:
        return []

    ptm_counts = collections.Counter(ptms_in_sequence)

    # Find all possible modification sites for each PTM type
    possible_sites_for_ptm: Dict[str, List[int]] = collections.defaultdict(list)
    for ptm_type, valid_aas in rules.items():
        if ptm_type in ptm_counts:
            for i, amino_acid in enumerate(bare_peptide):
                if amino_acid in valid_aas:
                    possible_sites_for_ptm[ptm_type].append(i)

    # Generate site combinations for each PTM type
    list_of_combinations_per_ptm = []
    ptm_order = list(ptm_counts.keys())

    for ptm_type in ptm_order:
        count = ptm_counts[ptm_type]
        sites = possible_sites_for_ptm[ptm_type]

        if len(sites) < count:
            return []

        combinations_for_this_ptm = list(itertools.combinations(sites, count))
        list_of_combinations_per_ptm.append(combinations_for_this_ptm)

    # Cartesian product of all PTM type combinations
    all_placement_scenarios = itertools.product(*list_of_combinations_per_ptm)

    # Build new sequences from each scenario
    result_sequences = []
    for scenario in all_placement_scenarios:
        ptm_placements: Dict[int, str] = {}
        valid_scenario = True
        for i, ptm_positions in enumerate(scenario):
            ptm_type = ptm_order[i]
            for pos in ptm_positions:
                if pos in ptm_placements:
                    valid_scenario = False
                    break
                ptm_placements[pos] = ptm_type
            if not valid_scenario:
                break

        if not valid_scenario:
            continue

        new_sequence = []
        for i, amino_acid in enumerate(bare_peptide):
            new_sequence.append(amino_acid)
            if i in ptm_placements:
                new_sequence.append(ptm_placements[i])

        if has_n_terminal_ac:
            new_sequence.insert(0, "(ac)")

        if new_sequence != original_sequence:
            result_sequences.append(new_sequence)

    if len(result_sequences) > 0:
        result_sequences = [''.join(r) for r in result_sequences]

    return result_sequences


def merge_tokens(tokens: List[str]) -> List[str]:
    """Merge PTM tokens with their preceding amino acid. e.g., ['S','(ph)','Y'] -> ['S(ph)','Y']"""
    merged = []
    for i, token in enumerate(tokens):
        if token in {"(ph)","(me)","(di)","(tr)","(ac)","(ox)","(ub)","(cam)","(deam)"}:
            if len(merged) > 0:
                merged[-1] = merged[-1] + token
            else:
                merged.append(token)
        else:
            merged.append(token)
    return merged

def compute_ptm_probabilities(base_seq: str,
                              base_score: float,
                              variations: List[List[str]],
                              variation_scores: List[float]) -> List[Dict[str, float]]:
    """Compute per-position PTM probabilities from sequence variants and their scores.

    For each PTM type, the probability at each position is computed as the sum of
    scores of all variants that place that PTM at that position, divided by the
    total score across all variants (global normalization per PTM type).
    """
    base_tokens = re.split(r"(?<=.)(?=[A-Z(])", base_seq)
    base_tokens = [t for t in base_tokens if t]
    base_seq_tokens = merge_tokens(base_tokens)

    has_n_terminal_ac = True if base_tokens[0] == "(ac)" else False

    seq_len = len(base_seq_tokens)
    all_variants = [base_seq_tokens] + [merge_tokens(re.split(r"(?<=.)(?=[A-Z(])", v)) for v in variations]
    all_scores = [base_score] + variation_scores

    # Accumulate score per (position, ptm) and total per ptm type
    pos_ptm_scores = [defaultdict(float) for _ in range(seq_len)]
    ptm_totals = defaultdict(float)

    for seq_tokens, score in zip(all_variants, all_scores):
        score_val = float(score)
        for pos, token in enumerate(seq_tokens):
            if has_n_terminal_ac and seq_tokens[0] != "(ac)":
                pos += 1
            elif has_n_terminal_ac and seq_tokens[0] == "(ac)" and pos == 0:
                pos_ptm_scores[pos]["(ac)"] += score_val
                ptm_totals["(ac)"] += score_val
                continue
            m = re.match(r"([A-Z]+)(\(.*\))?", token)
            if not m:
                continue
            aa, ptm = m.groups()[0], m.groups()[1]
            if ptm:
                pos_ptm_scores[pos][ptm] += score_val
                ptm_totals[ptm] += score_val

    # Normalize: each position's prob = its accumulated score / total for that PTM type
    probs = [defaultdict(float) for _ in range(seq_len)]
    for pos in range(seq_len):
        for ptm, score_sum in pos_ptm_scores[pos].items():
            if ptm_totals[ptm] > 0:
                probs[pos][ptm] = score_sum / ptm_totals[ptm]

    return probs

def reorder_label(x):
    """Reorder a label tensor so that non-negative values come first."""
    non_negative_mask = (x >= 0).to(x.device)
    count_non_negative = non_negative_mask.sum(dim=-1).to(x.device)

    result = torch.zeros_like(x).to(x.device)
    batch_size, seq_len = x.shape

    col_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(x.device)

    new_col_indices = torch.where(
        non_negative_mask,
        torch.cumsum(non_negative_mask, dim=1) - 1,
        count_non_negative.unsqueeze(1) + (col_indices - non_negative_mask.cumsum(dim=1))
    ).to(x.device)

    result.scatter_(1, new_col_indices, x)
    return result

def mass_cal(sequence, residues_raw):
    """Calculate total residue mass for a peptide sequence string."""
    residues = residues_raw.copy()
    sequence = sequence.replace("I", "L")
    sequence = re.split(r"(?<=.)(?=[A-Z(])", sequence)
    total = 0

    residues["_"] = 0
    for k, v in residues.items():
        if v < 0:
            residues[k] = 100000
    aa2mas = residues

    for each in sequence:
        try:
            total += aa2mas[each]
        except:
            h1 = each.count("+42.011")
            h2 = each.count("+43.006")
            h3 = each.count("-17.027")
            total += h1 * 42.010565 + h2 * 43.005814  + h3 * -17.026549
            each = each.replace("+42.011", "")
            each = each.replace("+43.006", "")
            each = each.replace("-17.027", "")
            if each:
                total += aa2mas[each]

    return total , sequence

def remove_repentance(index_list: List[int]) -> List[int]:
    """
    Eliminate repeated index in list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
    """
    return [a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b]

def ctc_post_processing(sentence_index: List[int]) -> List[int]:
    """
    Merge repetitive tokens, then eliminate blank tokens.
    The input sentence_index is expected to be a 1-D index list.
    """
    sentence_index = remove_repentance(sentence_index)
    temp = []
    for each in sentence_index:
        if each != 31:
            temp.append(each)
    return temp
