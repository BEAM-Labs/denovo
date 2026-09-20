#!/usr/bin/env python3
"""Model-aware adapters into a validated ProForma/Unimod sequence schema."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass



AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
RESIDUE_MONOISOTOPIC_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084063975,
    "L": 113.084063975,
    "K": 128.094963015,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}
WATER_MONOISOTOPIC_MASS = 18.010564684


@dataclass(frozen=True)
class Modification:
    name: str
    unimod_id: int | None
    delta_mass: float
    position: int | None
    site: str
    is_variable: bool
    source_token: str

    @property
    def proforma_tag(self) -> str:
        if self.unimod_id is not None:
            return f"[UNIMOD:{self.unimod_id}]"
        return f"[{self.delta_mass:+.6f}]"


@dataclass(frozen=True)
class ParsedPrediction:
    model: str
    raw_prediction: str
    is_empty: bool
    proforma: str
    backbone_strict: str
    backbone_il_equivalent: str
    modifications: tuple[Modification, ...]
    composition_key_strict: str
    composition_key_il_equivalent: str
    variable_site_key_strict: str
    variable_site_key_il_equivalent: str
    calculated_neutral_mass: float | None

    def as_dict(self) -> dict:
        result = asdict(self)
        result["modifications"] = [asdict(modification) for modification in self.modifications]
        return result


MODIFICATIONS = {
    "Carbamyl": (5, 43.005814, True),
    "Deamidated": (7, 0.984016, True),
    "Oxidation": (35, 15.994915, True),
    "Carbamidomethyl": (4, 57.021464, True),  # Explicit token, not a fixed background mass.
    "AmmoniaLoss": (None, -17.026549, True),
    "Acetyl": (1, 42.010565, True),
    "Dimethyl": (36, 28.031300, True),
    "Methyl": (34, 14.015650, True),
    "Trimethyl": (37, 42.046950, True),
    "GlyGly": (121, 114.042927, True),
    "Phospho": (21, 79.966331, True),
}

OMNI_TOKENS = {
    "(car)": "Carbamyl",
    "(deam)": "Deamidated",
    "(ox)": "Oxidation",
    "(cam)": "Carbamidomethyl",
    "(-nh3)": "AmmoniaLoss",
    "(ac)": "Acetyl",
    "(di)": "Dimethyl",
    "(me)": "Methyl",
    "(tr)": "Trimethyl",
    "(ub)": "GlyGly",
    "(ph)": "Phospho",
}


def modification_from_name(name: str, position: int | None, site: str, source_token: str) -> Modification:
    unimod_id, delta_mass, is_variable = MODIFICATIONS[name]
    return Modification(name, unimod_id, delta_mass, position, site, is_variable, source_token)


def tokenize(model: str, raw_prediction: str) -> tuple[list[str], list[Modification]]:
    residues: list[str] = []
    modifications: list[Modification] = []
    index = 0
    while index < len(raw_prediction):
        character = raw_prediction[index]
        if character in AMINO_ACIDS:
            residues.append(character)
            index += 1
            continue
        if model == "OmniNovo" and character == "(":
            end = raw_prediction.find(")", index + 1)
            if end == -1:
                raise ValueError(f"unterminated OmniNovo token at offset {index}")
            token = raw_prediction[index : end + 1]
            if token not in OMNI_TOKENS:
                raise ValueError(f"unknown OmniNovo modification token {token}")
            name = OMNI_TOKENS[token]
            if residues:
                position, site = len(residues), residues[-1]
            elif name in {"Acetyl", "Carbamyl", "AmmoniaLoss"}:
                position, site = None, "n_term"
            else:
                position, site = None, "unlocalized"
            modifications.append(modification_from_name(name, position, site, token))
            index = end + 1
            continue
        raise ValueError(f"invalid {model} character {character!r} at offset {index}")
    return residues, modifications


def build_proforma(residues: list[str], modifications: list[Modification]) -> str:
    unlocalized = "".join(
        modification.proforma_tag + "?" for modification in modifications if modification.site == "unlocalized"
    )
    n_terminal = "".join(
        modification.proforma_tag for modification in modifications if modification.site == "n_term"
    )
    sequence = []
    for position, residue in enumerate(residues, start=1):
        sequence.append(residue)
        sequence.extend(
            modification.proforma_tag
            for modification in modifications
            if modification.position == position and modification.site not in {"n_term", "unlocalized"}
        )
    n_terminal = n_terminal + "-" if n_terminal else ""
    return unlocalized + n_terminal + "".join(sequence)


def key_suffix(modifications: list[Modification], include_positions: bool) -> str:
    variable = [modification for modification in modifications if modification.is_variable]
    if include_positions:
        values = []
        for modification in variable:
            location = modification.position if modification.position is not None else modification.site
            values.append(f"{modification.name}@{location}")
        return ";".join(sorted(values))
    counts = Counter(modification.name for modification in variable)
    return ";".join(f"{name}:{counts[name]}" for name in sorted(counts))


def parse_prediction(model: str, raw_prediction: str | None) -> ParsedPrediction:
    if model != "OmniNovo":
        raise ValueError(f"unsupported model: {model}")
    raw = "" if raw_prediction is None else raw_prediction.strip()
    if not raw:
        return ParsedPrediction(model, raw, True, "", "", "", (), "", "", "", "", None)
    residues, modifications = tokenize(model, raw)
    if not residues:
        raise ValueError("prediction has no amino-acid residues")
    proforma = build_proforma(residues, modifications)
    backbone = "".join(residues)
    il_backbone = re.sub(r"[IL]", "J", backbone)
    composition_suffix = key_suffix(modifications, include_positions=False)
    site_suffix = key_suffix(modifications, include_positions=True)
    calculated_neutral_mass = (
        WATER_MONOISOTOPIC_MASS
        + sum(RESIDUE_MONOISOTOPIC_MASS[residue] for residue in residues)
        + sum(modification.delta_mass for modification in modifications)
    )
    return ParsedPrediction(
        model=model,
        raw_prediction=raw,
        is_empty=False,
        proforma=proforma,
        backbone_strict=backbone,
        backbone_il_equivalent=il_backbone,
        modifications=tuple(modifications),
        composition_key_strict=f"{backbone}|{composition_suffix}",
        composition_key_il_equivalent=f"{il_backbone}|{composition_suffix}",
        variable_site_key_strict=f"{backbone}|{site_suffix}",
        variable_site_key_il_equivalent=f"{il_backbone}|{site_suffix}",
        calculated_neutral_mass=calculated_neutral_mass,
    )
