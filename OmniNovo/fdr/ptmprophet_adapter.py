#!/usr/bin/env python3
"""Python adapter from structured de novo PSMs to PTMProphet and back."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
import shutil
import sys
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from sequence_schema import ParsedPrediction


PTMPROPHET = Path(os.environ.get("PTMPROPHET_BINARY", shutil.which("PTMProphetParser") or "PTMProphetParser"))
PYOPENMS_PYTHON = Path(os.environ.get("PYOPENMS_PYTHON", sys.executable))
CONVERTER = Path(__file__).with_name("convert_mgf_to_indexed_mzml.py")
PROTON = 1.007276466621
NTERM_H = 1.00782503223
RESIDUE_MASS = {
    "A": 71.037114, "R": 156.101111, "N": 114.042927, "D": 115.026943,
    "C": 103.009185, "E": 129.042593, "Q": 128.058578, "G": 57.021464,
    "H": 137.058912, "I": 113.084064, "L": 113.084064, "K": 128.094963,
    "M": 131.040485, "F": 147.068414, "P": 97.052764, "S": 87.032028,
    "T": 101.047670, "W": 186.079313, "Y": 163.063329, "V": 99.068414,
}
PTMPROPHET_OPTIONS = (
    "EM=2",
    "MINPROB=0",
    "STATIC",
    "FRAGPPMTOL=20",
    "MAXTHREADS=16",
    "NIONS=b",
    "CIONS=y",
    "MAXFRAGZ=-1",
    "NOSTACK",
    "VERBOSE",
    "STY:79.966331",
)
PTMPROPHET_DIRECT_OPTIONS = (
    "EM=0",
    "DIRECT",
    "MINPROB=0",
    "STATIC",
    "FRAGPPMTOL=20",
    "MAXTHREADS=16",
    "NIONS=b",
    "CIONS=y",
    "MAXFRAGZ=-1",
    "NOSTACK",
    "VERBOSE",
    "STY:79.966331",
)
PTMPROPHET_EM0_OPTIONS = tuple(
    option for option in PTMPROPHET_DIRECT_OPTIONS if option != "DIRECT"
)
TARGET_MASS_TOLERANCE = 0.005  # MODPREC=4; separates Acetyl and Trimethyl.
# PTMProphet takes all modification specifications as ONE comma-separated
# argument. Passing them as separate argv tokens makes the last one win and the
# rest are silently dropped.
_SINGLE_SPEC = r"[A-Znc]+:-?\d+(?:\.\d+)?(?::-?\d+(?:\.\d+)?)*"
MODIFICATION_SPEC = re.compile(rf"^{_SINGLE_SPEC}(?:,{_SINGLE_SPEC})*$")


@dataclass(frozen=True)
class LocalizationTarget:
    """One modification whose position PTMProphet is asked to localize.

    ``name`` must match the ``ParsedPrediction`` modification name so the
    adapter can tell targeted from untargeted modifications. ``option`` is the
    PTMProphet modification specification, which the engine echoes verbatim as
    the ``ptm`` attribute of each ``ptmprophet_result``; that echo is how the
    parser assigns site probabilities back to their modification.
    """

    name: str
    residues: str
    delta_mass: float
    neutral_losses: tuple[float, ...] = ()

    @property
    def option(self) -> str:
        parts = [self.residues, f"{self.delta_mass:.6f}"]
        parts.extend(f"{loss:.6f}" for loss in self.neutral_losses)
        return ":".join(parts)


PHOSPHO_STY = LocalizationTarget("Phospho", "STY", 79.966331)
DEFAULT_TARGETS = (PHOSPHO_STY,)


def validate_targets(targets: tuple[LocalizationTarget, ...]) -> None:
    if not targets:
        raise ValueError("at least one localization target is required")
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError(f"localization target names must be unique: {names}")
    options = [target.option for target in targets]
    if len(set(options)) != len(options):
        raise ValueError(f"localization target specifications must be unique: {options}")
    for target in targets:
        unknown = sorted(set(target.residues) - (set(RESIDUE_MASS) | {"n"}))
        if unknown:
            raise ValueError(
                f"target {target.name} has residues this adapter cannot place: {unknown}; "
                "only residues and the peptide N terminus (n) are supported"
            )
    for index, first in enumerate(targets):
        for second in targets[index + 1 :]:
            if set(first.residues) & set(second.residues) and abs(
                first.delta_mass - second.delta_mass
            ) <= 2 * TARGET_MASS_TOLERANCE:
                raise ValueError(
                    f"targets {first.name} and {second.name} share residues and are not "
                    f"separable at +/-{TARGET_MASS_TOLERANCE} Da"
                )


def modification_specs(options: tuple[str, ...]) -> tuple[str, ...]:
    """Return every modification specification declared in ``options``, one per target."""
    return tuple(
        spec
        for option in options
        if MODIFICATION_SPEC.match(option)
        for spec in option.split(",")
    )


def with_targets(
    base_options: tuple[str, ...], targets: tuple[LocalizationTarget, ...]
) -> tuple[str, ...]:
    """Return ``base_options`` with its modification specifications replaced."""
    validate_targets(targets)
    kept = tuple(option for option in base_options if not MODIFICATION_SPEC.match(option))
    return (*kept, ",".join(target.option for target in targets))


def check_options_match_targets(
    options: tuple[str, ...],
    targets: tuple[LocalizationTarget, ...],
    auxiliary_specs: tuple[str, ...] = (),
) -> None:
    expected = (*auxiliary_specs, *(target.option for target in targets))
    found = modification_specs(options)
    if found != expected:
        raise ValueError(
            f"PTMProphet options declare modifications {found} but the adapter was given "
            f"targets {expected}; build options with with_targets()"
        )


def pepxml_residue_modification_is_variable(residue: str, modification) -> bool:
    """Keep every de novo nonphospho modification PSM-specific.

    Unimod's general fixed/variable classification is not a pepXML population
    declaration. For example, a de novo S[+57] prediction must remain a
    PSM-specific variable modification; declaring it fixed would add +57 to
    every serine in the population. The normalized de novo schema also treats
    a bare C at its unmodified mass, so even explicit C[+57] is PSM-specific.
    """
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_mgf(path: Path):
    metadata: dict[str, str] = {}
    peaks: list[str] = []
    active = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                metadata, peaks, active = {}, [], True
            elif line == "END IONS":
                yield metadata, peaks
                active = False
            elif not active or not line:
                continue
            elif "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
            else:
                peaks.append(line)
    if active:
        raise RuntimeError(f"unterminated MGF spectrum in {path}")


def write_selected_mgf(source: Path, selected_indices: list[int], output: Path) -> None:
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("PTMProphet input has duplicate source spectrum indices")
    if selected_indices != sorted(selected_indices):
        raise ValueError("PTMProphet input PSMs must be sorted by source spectrum index")
    index_to_scan = {source_index: adapter_scan for adapter_scan, source_index in enumerate(selected_indices, start=1)}
    found = set()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        # TPP 6.3.2 indexes mzML spectra by a positive, zero-based vector offset.
        # A one-peak unreferenced sentinel at offset 0 makes adapter scans 1..N valid.
        handle.write("BEGIN IONS\nTITLE=ptmprophet_unreferenced_sentinel\nPEPMASS=500\nCHARGE=2+\nSCANS=0\n100.00000 1.00000\nEND IONS\n\n")
        for source_index, (metadata, peaks) in enumerate(iter_mgf(source)):
            if source_index not in index_to_scan:
                continue
            adapter_scan = index_to_scan[source_index]
            found.add(source_index)
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE=ptmprophet_scan={adapter_scan}|source_index={source_index}\n")
            handle.write(f"PEPMASS={metadata['PEPMASS']}\n")
            handle.write(f"CHARGE={metadata['CHARGE']}\n")
            handle.write(f"SCANS={adapter_scan}\n")
            for peak in peaks:
                handle.write(peak + "\n")
            handle.write("END IONS\n\n")
            if len(found) == len(selected_indices):
                break
    missing = set(selected_indices) - found
    if missing:
        raise RuntimeError(f"missing {len(missing)} requested spectra from {source}: {sorted(missing)[:10]}")


def match_groups_to_slots(candidates: list[list[int]]) -> list[int] | None:
    """Assign each pending group a distinct free position, or return None.

    Greedy assignment in target order is not enough once targets share residues:
    with targets Methyl(KR) and Acetyl(K) on a peptide holding one K and one R,
    taking K for Methyl first strands Acetyl even though Methyl(R) + Acetyl(K)
    is feasible. This is Kuhn's augmenting-path matching; candidate lists are
    kept in ascending order so the assignment stays deterministic.
    """
    assignment: dict[int, int] = {}

    def assign(group: int, seen: set[int]) -> bool:
        for position in candidates[group]:
            if position in seen:
                continue
            seen.add(position)
            if position not in assignment or assign(assignment[position], seen):
                assignment[position] = group
                return True
        return False

    for group in range(len(candidates)):
        if not assign(group, set()):
            return None
    result = [-1] * len(candidates)
    for position, group in assignment.items():
        result[group] = position
    return result


def initial_modification_positions(
    parsed: ParsedPrediction, targets: tuple[LocalizationTarget, ...] = DEFAULT_TARGETS
) -> tuple[list[tuple[int, object]], list[object]]:
    """Return residue and N-terminal mods, assigning invalid/unlocalized targets deterministically."""
    backbone = parsed.backbone_strict
    by_name = {target.name: target for target in targets}
    compatible = {t.name: ([0] if "n" in t.residues else []) + [
        i for i, aa in enumerate(backbone, 1) if aa in t.residues
    ] for t in targets}
    groups = list(parsed.modifications)
    candidates = []
    for mod in groups:
        original = 0 if mod.site == "n_term" else mod.position
        if mod.name in by_name:
            slots = compatible[mod.name]
            # Keep the predicted site first when valid, then initialize otherwise.
            candidates.append(([original] if original in slots else []) + [i for i in slots if i != original])
        elif original is not None:
            candidates.append([original])
        else:
            raise ValueError(f"cannot initialize untargeted unlocalized modification {mod.name}")
    assigned = match_groups_to_slots(candidates)
    if assigned is None:
        raise ValueError("no feasible assignment of modification groups to distinct sites")
    residue_mods = [(pos, mod) for pos, mod in zip(assigned, groups) if pos != 0]
    terminal_mods = [mod for pos, mod in zip(assigned, groups) if pos == 0]
    return residue_mods, terminal_mods


LOCALIZATION_EXCLUSIONS = (
    "no_targeted_modification",
    "insufficient_target_residue_capacity",
    "unlocalized_untargeted_modification",
    "stacked_residue_modifications_incompatible_with_NOSTACK",
    "untargeted_modification_on_target_residue",
    "no_feasible_target_assignment",
)


def localization_exclusion_reason(
    parsed: ParsedPrediction, targets: tuple[LocalizationTarget, ...] = DEFAULT_TARGETS
) -> str | None:
    """Return why this prediction cannot be localized, or None if it is eligible.

    ``no_targeted_modification`` means there is nothing for PTMProphet to place
    — a different situation from the remaining reasons, which are predictions
    the adapter or the NOSTACK engine mode cannot represent. Callers deciding
    what to do with an unmodified PSM should branch on that reason specifically.

    The rule order reproduces the phospho-only filter this generalizes, so the
    per-category counts of a single-target run are unchanged.
    """
    backbone = parsed.backbone_strict
    target_names = {target.name for target in targets}
    target_residues = set().union(*(set(target.residues) for target in targets))

    counts = {target.name: 0 for target in targets}
    for modification in parsed.modifications:
        if modification.name in target_names:
            counts[modification.name] += 1
    if not sum(counts.values()):
        return "no_targeted_modification"
    for target in targets:
        capacity = sum(residue in target.residues for residue in backbone) + ("n" in target.residues)
        if counts[target.name] > capacity:
            return "insufficient_target_residue_capacity"
    if any(
        modification.name not in target_names and modification.site == "unlocalized"
        for modification in parsed.modifications
    ):
        return "unlocalized_untargeted_modification"
    positions = [
        0 if modification.site == "n_term" else modification.position
        for modification in parsed.modifications
        if modification.position is not None or modification.site == "n_term"
    ]
    if len(positions) != len(set(positions)):
        return "stacked_residue_modifications_incompatible_with_NOSTACK"
    if any(
        modification.name not in target_names
        and modification.position is not None
        and backbone[modification.position - 1] in target_residues
        for modification in parsed.modifications
    ):
        return "untargeted_modification_on_target_residue"
    try:
        residue_mods, terminal_mods = initial_modification_positions(parsed, targets)
    except ValueError:
        return "no_feasible_target_assignment"
    return None


def write_pepxml(
    psms: list[dict],
    mzml_path: Path,
    output: Path,
    eligibility_probability: float = 1.0,
    targets: tuple[LocalizationTarget, ...] = DEFAULT_TARGETS,
) -> list[dict]:
    validate_targets(targets)
    target_names = {target.name for target in targets}
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    namespace = "http://regis-web.systemsbiology.net/pepXML"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("", namespace)
    ET.register_namespace("xsi", xsi)

    def element(parent, tag, attributes=None):
        return ET.SubElement(parent, f"{{{namespace}}}{tag}", attributes or {})

    root = ET.Element(
        f"{{{namespace}}}msms_pipeline_analysis",
        {
            "date": generated_at,
            "summary_xml": str(output),
            f"{{{xsi}}}schemaLocation": f"{namespace} http://sashimi.sourceforge.net/schema_revision/pepXML/pepXML_v122.xsd",
        },
    )
    analysis = element(root, "analysis_summary", {"analysis": "peptideprophet", "time": generated_at})
    prophet_summary = element(analysis, "peptideprophet_summary", {"min_prob": "0.0"})
    element(prophet_summary, "inputfile", {"name": str(output)})
    element(root, "dataset_derivation", {"generation_no": "0"})
    base_name = str(mzml_path.with_suffix(""))
    run = element(
        root,
        "msms_run_summary",
        {"base_name": base_name, "raw_data_type": "mzML", "raw_data": "mzML"},
    )
    enzyme = element(run, "sample_enzyme", {"name": "nonspecific"})
    element(enzyme, "specificity", {"cut": "", "no_cut": "", "sense": "C"})
    search = element(
        run,
        "search_summary",
        {
            "base_name": base_name,
            "precursor_mass_type": "monoisotopic",
            "search_engine": "de_novo_adapter",
            "search_engine_version": "1",
            "fragment_mass_type": "monoisotopic",
            "search_id": "1",
        },
    )
    element(search, "search_database", {"local_path": "de_novo_no_database.fasta", "type": "AA"})
    element(search, "enzymatic_search_constraint", {"enzyme": "nonspecific", "min_number_termini": "0", "max_num_internal_cleavages": "100"})
    residue_declarations = set()
    terminal_declarations = set()
    for psm in psms:
        parsed: ParsedPrediction = psm["parsed_prediction"]
        residue_mods, terminal_mods = initial_modification_positions(parsed, targets)
        for position, modification in residue_mods:
            if modification.name not in target_names:
                residue_declarations.add(
                    (
                        parsed.backbone_strict[position - 1],
                        modification.delta_mass,
                        pepxml_residue_modification_is_variable(
                            parsed.backbone_strict[position - 1], modification
                        ),
                    )
                )
        for modification in terminal_mods:
            terminal_declarations.add((modification.delta_mass, True))
    for target in targets:
        if "n" in target.residues:
            terminal_declarations.add((target.delta_mass, True))
        for residue in target.residues.replace("n", ""):
            element(
                search,
                "aminoacid_modification",
                {
                    "aminoacid": residue,
                    "massdiff": f"{target.delta_mass:.6f}",
                    "mass": f"{RESIDUE_MASS[residue] + target.delta_mass:.6f}",
                    "variable": "Y",
                },
            )
    for residue, delta_mass, is_variable in sorted(residue_declarations):
        element(
            search,
            "aminoacid_modification",
            {
                "aminoacid": residue,
                "massdiff": f"{delta_mass:.6f}",
                "mass": f"{RESIDUE_MASS[residue] + delta_mass:.6f}",
                "variable": "Y" if is_variable else "N",
            },
        )
    for delta_mass, is_variable in sorted(terminal_declarations):
        element(
            search,
            "terminal_modification",
            {
                "terminus": "n",
                "massdiff": f"{delta_mass:.6f}",
                "mass": f"{NTERM_H + delta_mass:.6f}",
                "variable": "Y" if is_variable else "N",
                "protein_terminus": "N",
            },
        )
    element(search, "parameter", {"name": "fragment_ion_series", "value": "b,y"})
    element(search, "parameter", {"name": "fragment_mass_tolerance", "value": "20"})
    element(search, "parameter", {"name": "fragment_mass_units", "value": "1"})
    element(search, "parameter", {"name": "adapter_probability_semantics", "value": "eligibility_constant_not_model_probability"})

    mapping = []
    for adapter_scan, psm in enumerate(psms, start=1):
        mzml_scan = adapter_scan
        parsed: ParsedPrediction = psm["parsed_prediction"]
        residue_mods, terminal_mods = initial_modification_positions(parsed, targets)
        observed_mass = (float(psm["precursor_mz"]) - PROTON) * int(psm["precursor_charge"])
        calculated_mass = float(parsed.calculated_neutral_mass)
        spectrum_name = f"ptmprophet_input.{adapter_scan}.{adapter_scan}.{int(psm['precursor_charge'])}"
        query = element(
            run,
            "spectrum_query",
            {
                "start_scan": str(adapter_scan),
                "end_scan": str(adapter_scan),
                "index": str(adapter_scan),
                "spectrum": spectrum_name,
                "spectrumNativeID": f"scan={adapter_scan}",
                "assumed_charge": str(int(psm["precursor_charge"])),
                "precursor_neutral_mass": f"{observed_mass:.8f}",
                "uncalibrated_precursor_neutral_mass": f"{observed_mass:.8f}",
                "retention_time_sec": f"{float(adapter_scan):.1f}",
            },
        )
        result = element(query, "search_result")
        hit = element(
            result,
            "search_hit",
            {
                "hit_rank": "1",
                "peptide": parsed.backbone_strict,
                "peptide_prev_aa": "-",
                "peptide_next_aa": "-",
                "protein": "DE_NOVO_CANDIDATE",
                "protein_descr": "de novo candidate; no database mapping",
                "num_tot_proteins": "1",
                "num_matched_ions": "0",
                "tot_num_ions": str(max(0, 2 * (len(parsed.backbone_strict) - 1))),
                "calc_neutral_pep_mass": f"{calculated_mass:.8f}",
                "massdiff": f"{observed_mass - calculated_mass:.8f}",
                "num_tol_term": "0",
                "num_missed_cleavages": "0",
                "is_rejected": "0",
            },
        )
        if residue_mods or terminal_mods:
            modified = list(parsed.backbone_strict)
            for position, modification in sorted(residue_mods, reverse=True):
                absolute_mass = RESIDUE_MASS[parsed.backbone_strict[position - 1]] + modification.delta_mass
                modified.insert(position, f"[{absolute_mass:.6f}]")
            attributes = {"modified_peptide": "".join(modified)}
            if terminal_mods:
                attributes["mod_nterm_mass"] = f"{NTERM_H + sum(mod.delta_mass for mod in terminal_mods):.6f}"
            modification_info = element(hit, "modification_info", attributes)
            for position, modification in sorted(residue_mods):
                absolute_mass = RESIDUE_MASS[parsed.backbone_strict[position - 1]] + modification.delta_mass
                element(modification_info, "mod_aminoacid_mass", {"position": str(position), "mass": f"{absolute_mass:.6f}"})
        element(hit, "search_score", {"name": "native_score", "value": f"{float(psm['native_score']):.12g}"})
        element(
            hit,
            "search_score",
            {"name": "adapter_eligibility_constant", "value": f"{eligibility_probability:.6g}"},
        )
        analysis_result = element(hit, "analysis_result", {"analysis": "peptideprophet"})
        element(
            analysis_result,
            "peptideprophet_result",
            {
                "probability": f"{eligibility_probability:.6g}",
                "all_ntt_prob": f"({eligibility_probability:.6g},{eligibility_probability:.6g},{eligibility_probability:.6g})",
            },
        )
        initial_target_sites = {
            target.name: sorted(
                position
                for position, modification in [*residue_mods, *((0, m) for m in terminal_mods)]
                if modification.name == target.name
            )
            for target in targets
        }
        mapping.append(
            {
                "adapter_scan": adapter_scan,
                "mzml_scan": mzml_scan,
                "spectrum_index": int(psm["spectrum_index"]),
                "spectrum_id": str(psm["spectrum_id"]),
                "model": str(psm["model"]),
                "initial_proforma": parsed.proforma,
                "initial_backbone": parsed.backbone_strict,
                "initial_target_sites": initial_target_sites,
                # Retained under its original name for the phospho-only sessions that
                # already consume this column; empty when Phospho is not a target.
                "initial_phospho_sites": [position for position, modification in residue_mods if modification.name == "Phospho"],
                "initial_nonphospho_modifications": [
                    f"{position}:{modification.name}:{modification.delta_mass:.6f}"
                    for position, modification in residue_mods
                    if modification.name not in target_names
                ]
                + [f"n_term:{modification.name}:{modification.delta_mass:.6f}" for modification in terminal_mods if modification.name not in target_names],
            }
        )
    ET.ElementTree(root).write(output, encoding="UTF-8", xml_declaration=True)
    return mapping


def serialize_mapping_row(row: dict) -> dict:
    return {
        **row,
        "initial_target_sites": json.dumps(
            {name: list(sites) for name, sites in row["initial_target_sites"].items()},
            sort_keys=True,
        ),
        "initial_phospho_sites": ",".join(map(str, row["initial_phospho_sites"])),
        "initial_nonphospho_modifications": ";".join(row["initial_nonphospho_modifications"]),
    }


def run_ptmprophet(
    psms: Iterable[dict],
    source_mgf: Path,
    output_dir: Path,
    options: tuple[str, ...] = PTMPROPHET_OPTIONS,
    eligibility_probability: float = 1.0,
    targets: tuple[LocalizationTarget, ...] = DEFAULT_TARGETS,
) -> dict:
    check_options_match_targets(options, targets)
    psm_list = list(psms)
    if not psm_list:
        raise ValueError("PTMProphet adapter received no PSMs")
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_mgf = output_dir / "selected_spectra.mgf"
    mzml = output_dir / "ptmprophet_input.mzML"
    input_pepxml = output_dir / "ptmprophet_input.pep.xml"
    output_pepxml = output_dir / "ptmprophet_output.pep.xml"
    mapping_path = output_dir / "scan_mapping.tsv"
    write_selected_mgf(source_mgf, [int(psm["spectrum_index"]) for psm in psm_list], selected_mgf)
    convert = subprocess.run(
        [str(PYOPENMS_PYTHON), str(CONVERTER), str(selected_mgf), str(mzml)],
        check=True,
        text=True,
        capture_output=True,
    )
    if not 0 < eligibility_probability <= 1:
        raise ValueError("eligibility_probability must be in (0, 1]")
    mapping = write_pepxml(psm_list, mzml, input_pepxml, eligibility_probability, targets)
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in mapping:
            writer.writerow(serialize_mapping_row(row))
    command = [str(PTMPROPHET), *options, str(input_pepxml), str(output_pepxml)]
    process = subprocess.run(command, check=False, text=True, capture_output=True)
    (output_dir / "ptmprophet.stdout.log").write_text(process.stdout, encoding="utf-8")
    (output_dir / "ptmprophet.stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0 or not output_pepxml.exists():
        raise RuntimeError(
            f"PTMProphet failed with exit {process.returncode}; see {output_dir / 'ptmprophet.stderr.log'}"
        )
    manifest = {
        "status": "completed",
        "ptmprophet_binary": str(PTMPROPHET),
        "ptmprophet_binary_sha256": sha256(PTMPROPHET),
        "options": list(options),
        "localization_targets": [target.option for target in targets],
        "eligibility_probability": eligibility_probability,
        "eligibility_probability_semantics": "technical inclusion constant, not a model posterior or q-value",
        "psms": len(psm_list),
        "source_mgf": str(source_mgf),
        "source_mgf_sha256": sha256(source_mgf),
        "converter_stdout": convert.stdout.strip(),
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (selected_mgf, mzml, input_pepxml, output_pepxml, mapping_path)
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "mapping": mapping, "output_pepxml": output_pepxml}


def run_ptmprophet_sharded_direct(
    psms: Iterable[dict],
    source_mgf: Path,
    output_dir: Path,
    shards: int = 16,
    max_psms_per_shard: int | None = None,
    options: tuple[str, ...] = PTMPROPHET_DIRECT_OPTIONS,
    eligibility_probability: float = 1.0,
    targets: tuple[LocalizationTarget, ...] = DEFAULT_TARGETS,
) -> dict:
    """Run independent EM=0 PSM shards against one shared mzML.

    EM=0 has no iterative population fit. Each bounded shard can therefore be
    processed independently. The caller supplies either the diagnostic DIRECT
    profile or the formal non-DIRECT EM=0 profile. The outer process count
    supplies local CPU parallelism; each official PTMProphet process is capped
    at one internal worker so total active workers do not exceed ``shards``.
    """
    check_options_match_targets(options, targets)
    if "EM=0" not in options:
        raise ValueError("sharded execution is restricted to EM=0")
    if shards < 2 or shards > 16:
        raise ValueError("shards must be between 2 and 16")
    if not 0 < eligibility_probability <= 1:
        raise ValueError("eligibility_probability must be in (0, 1]")
    psm_list = list(psms)
    if max_psms_per_shard is not None and max_psms_per_shard < 1:
        raise ValueError("max_psms_per_shard must be positive")
    if max_psms_per_shard is None and len(psm_list) < shards:
        raise ValueError("PTMProphet sharding requires at least one PSM per shard")
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_mgf = output_dir / "selected_spectra.mgf"
    mzml = output_dir / "ptmprophet_input.mzML"
    input_pepxml = output_dir / "ptmprophet_input.pep.xml"
    mapping_path = output_dir / "scan_mapping.tsv"
    write_selected_mgf(
        source_mgf,
        [int(psm["spectrum_index"]) for psm in psm_list],
        selected_mgf,
    )
    convert = subprocess.run(
        [str(PYOPENMS_PYTHON), str(CONVERTER), str(selected_mgf), str(mzml)],
        check=True,
        text=True,
        capture_output=True,
    )
    mapping = write_pepxml(psm_list, mzml, input_pepxml, eligibility_probability, targets)
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(mapping[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in mapping:
            writer.writerow(serialize_mapping_row(row))

    namespace = "http://regis-web.systemsbiology.net/pepXML"
    tree = ET.parse(input_pepxml)
    root = tree.getroot()
    run = root.find(f"{{{namespace}}}msms_run_summary")
    if run is None:
        raise RuntimeError("adapter pepXML has no msms_run_summary")
    queries = run.findall(f"{{{namespace}}}spectrum_query")
    if len(queries) != len(mapping):
        raise RuntimeError("adapter pepXML/query mapping coverage mismatch")
    for query in queries:
        run.remove(query)

    shard_root = output_dir / "direct_shards"
    shard_root.mkdir()
    mapping_by_scan = {int(row["adapter_scan"]): row for row in mapping}
    if max_psms_per_shard is None:
        query_groups = [queries[index::shards] for index in range(shards)]
    else:
        query_groups = [
            queries[start : start + max_psms_per_shard]
            for start in range(0, len(queries), max_psms_per_shard)
        ]
    if len(query_groups) < 2:
        raise ValueError("sharded execution requires at least two query groups")
    shard_records = []
    for shard_index, shard_queries in enumerate(query_groups):
        directory = shard_root / f"shard-{shard_index:02d}"
        directory.mkdir()
        shard_xml_root = copy.deepcopy(root)
        shard_run = shard_xml_root.find(f"{{{namespace}}}msms_run_summary")
        assert shard_run is not None
        for query in shard_queries:
            shard_run.append(query)
        shard_input = directory / "input.pep.xml"
        shard_output = directory / "output.pep.xml"
        ET.ElementTree(shard_xml_root).write(
            shard_input, encoding="UTF-8", xml_declaration=True
        )
        shard_mapping = [
            mapping_by_scan[int(query.attrib["start_scan"])] for query in shard_queries
        ]
        shard_records.append(
            {
                "index": shard_index,
                "directory": directory,
                "input": shard_input,
                "output": shard_output,
                "mapping": shard_mapping,
            }
        )

    process_options = tuple(
        "MAXTHREADS=1" if option.startswith("MAXTHREADS=") else option
        for option in options
    )

    def execute(record: dict) -> dict:
        command = [
            str(PTMPROPHET),
            *process_options,
            str(record["input"]),
            str(record["output"]),
        ]
        process = subprocess.run(command, check=False, text=True, capture_output=True)
        (record["directory"] / "stdout.log").write_text(
            process.stdout, encoding="utf-8"
        )
        (record["directory"] / "stderr.log").write_text(
            process.stderr, encoding="utf-8"
        )
        if process.returncode != 0 or not record["output"].is_file():
            raise RuntimeError(
                f"PTMProphet shard {record['index']} failed with exit {process.returncode}"
            )
        return record

    with ThreadPoolExecutor(max_workers=shards) as executor:
        completed = list(executor.map(execute, shard_records))
    parsed_rows = []
    for record in completed:
        parsed_rows.extend(
            parse_ptmprophet_output(record["output"], record["mapping"], targets)
        )
    parsed_rows.sort(key=lambda row: int(row["adapter_scan"]))
    if len(parsed_rows) != len(mapping):
        raise RuntimeError("sharded PTMProphet parsed output coverage mismatch")

    manifest = {
        "status": "completed",
        "execution": "independent_psm_shards_for_EM0",
        "ptmprophet_binary": str(PTMPROPHET),
        "ptmprophet_binary_sha256": sha256(PTMPROPHET),
        "formal_options": list(options),
        "localization_targets": [target.option for target in targets],
        "per_process_options": list(process_options),
        "shards": len(shard_records),
        "max_psms_per_shard": max_psms_per_shard,
        "maximum_concurrent_official_processes": min(shards, len(shard_records)),
        "eligibility_probability": eligibility_probability,
        "eligibility_probability_semantics": (
            "technical inclusion constant, not a model posterior or q-value"
        ),
        "psms": len(psm_list),
        "source_mgf": str(source_mgf),
        "source_mgf_sha256": sha256(source_mgf),
        "converter_stdout": convert.stdout.strip(),
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (selected_mgf, mzml, input_pepxml, mapping_path)
        },
        "shard_outputs": [
            {
                "index": record["index"],
                "psms": len(record["mapping"]),
                "size_bytes": record["output"].stat().st_size,
                "sha256": sha256(record["output"]),
            }
            for record in completed
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "mapping": mapping, "parsed_rows": parsed_rows}


def match_target(
    residue: str, delta_mass: float, targets: tuple[LocalizationTarget, ...]
) -> LocalizationTarget | None:
    matches = [
        target
        for target in targets
        if residue in target.residues
        and abs(delta_mass - target.delta_mass) <= TARGET_MASS_TOLERANCE
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"{residue}{delta_mass:+.6f} matches several localization targets: "
            f"{[target.name for target in matches]}"
        )
    return matches[0] if matches else None


def parse_ptmprophet_output(
    output_pepxml: Path,
    mapping: list[dict],
    targets: tuple[LocalizationTarget, ...] = DEFAULT_TARGETS,
    auxiliary_specs: tuple[str, ...] = (),
) -> list[dict]:
    """Parse PTMProphet site probabilities and restore original spectrum identifiers.

    PTMProphet emits one ``ptmprophet_result`` per modification specification,
    identified by its ``ptm`` attribute. Site probabilities are grouped by that
    attribute so each target is scored against its own candidate sites.
    """
    validate_targets(targets)
    namespace = {"p": "http://regis-web.systemsbiology.net/pepXML"}
    option_to_target = {target.option: target for target in targets}
    auxiliary_spec_set = set(auxiliary_specs)
    mapping_by_scan = {int(row["adapter_scan"]): row for row in mapping}
    root = ET.parse(output_pepxml).getroot()
    rows = []
    for query in root.findall(".//p:spectrum_query", namespace):
        adapter_scan = int(query.attrib["start_scan"])
        if adapter_scan not in mapping_by_scan:
            raise RuntimeError(f"unexpected PTMProphet adapter scan {adapter_scan}")
        mapping_row = mapping_by_scan[adapter_scan]
        hit = query.find("p:search_result/p:search_hit", namespace)
        if hit is None:
            raise RuntimeError(f"adapter scan {adapter_scan} has no search hit")
        peptide = hit.attrib["peptide"]
        ptm_results = {}
        for analysis_result in hit.findall("p:analysis_result", namespace):
            if analysis_result.attrib.get("analysis") != "ptmprophet":
                continue
            for ptm_result in analysis_result.findall("p:ptmprophet_result", namespace):
                option = ptm_result.attrib["ptm"]
                if option in ptm_results:
                    raise RuntimeError(
                        f"adapter scan {adapter_scan} has duplicate PTMProphet results for {option}"
                    )
                ptm_results[option] = ptm_result
        if not ptm_results:
            raise RuntimeError(f"adapter scan {adapter_scan} has no PTMProphet result")
        unexpected = sorted(
            set(ptm_results) - set(option_to_target) - auxiliary_spec_set
        )
        if unexpected:
            raise RuntimeError(
                f"adapter scan {adapter_scan} reports unrequested modifications {unexpected}"
            )

        auxiliary_results = {}
        for option in auxiliary_specs:
            auxiliary_result = ptm_results.get(option)
            if auxiliary_result is None:
                continue
            parameters = {
                parameter.attrib["name"]: parameter.attrib["value"]
                for parameter in auxiliary_result.findall("p:parameter", namespace)
            }
            auxiliary_results[option] = {
                "ptm_peptide": auxiliary_result.attrib["ptm_peptide"],
                "mean_best_prob": float(parameters.get("mean_best_prob", "nan")),
                "terminal_probabilities": [
                    {
                        "terminus": site.attrib["terminus"],
                        "probability": float(site.attrib["probability"]),
                    }
                    for site in auxiliary_result.findall(
                        "p:mod_terminal_probability", namespace
                    )
                ],
                "residue_probabilities": [
                    {
                        "position": int(site.attrib["position"]),
                        "probability": float(site.attrib["probability"]),
                    }
                    for site in auxiliary_result.findall(
                        "p:mod_aminoacid_probability", namespace
                    )
                ],
            }

        initial_target_sites = mapping_row["initial_target_sites"]
        sites_by_target = {}
        probabilities_by_target = {}
        mbpr_by_target = {}
        ptm_peptide_by_target = {}
        mean_best_prob_by_target = {}
        norm_info_gain_by_target = {}
        localized_site_count = 0
        best_probability_sum = 0.0
        for target in targets:
            expected = len(initial_target_sites.get(target.name, ()))
            ptm_result = ptm_results.get(target.option)
            if ptm_result is None:
                if expected:
                    raise RuntimeError(
                        f"adapter scan {adapter_scan} carries {expected} {target.name} groups "
                        f"but PTMProphet reported no {target.option} result"
                    )
                sites_by_target[target.name] = []
                probabilities_by_target[target.name] = []
                mbpr_by_target[target.name] = None
                ptm_peptide_by_target[target.name] = None
                mean_best_prob_by_target[target.name] = float("nan")
                norm_info_gain_by_target[target.name] = float("nan")
                continue
            sites = [
                {
                    "position": int(site.attrib["position"]),
                    "residue": peptide[int(site.attrib["position"]) - 1],
                    "probability": float(site.attrib["probability"]),
                }
                for site in ptm_result.findall("p:mod_aminoacid_probability", namespace)
            ]
            sites.extend({"position": 0, "residue": "n", "probability": float(site.attrib["probability"])}
                         for site in ptm_result.findall("p:mod_terminal_probability", namespace)
                         if site.attrib["terminus"] == "n")
            if any(not math.isfinite(site["probability"]) or not 0 <= site["probability"] <= 1 for site in sites):
                raise RuntimeError(f"invalid PTMProphet probability at scan {adapter_scan}")
            if len({site["position"] for site in sites}) != len(sites):
                raise RuntimeError(f"duplicate PTMProphet sites at scan {adapter_scan}")
            ordered = sorted((site["probability"] for site in sites), reverse=True)
            if len(ordered) < expected:
                raise RuntimeError(
                    f"adapter scan {adapter_scan} has fewer {target.name} site probabilities "
                    f"({len(ordered)}) than {target.name} groups ({expected})"
                )
            parameters = {
                parameter.attrib["name"]: parameter.attrib["value"]
                for parameter in ptm_result.findall("p:parameter", namespace)
            }
            sites_by_target[target.name] = sites
            probabilities_by_target[target.name] = ordered
            mbpr_by_target[target.name] = (
                sum(ordered[:expected]) / expected if expected else None
            )
            ptm_peptide_by_target[target.name] = ptm_result.attrib["ptm_peptide"]
            mean_best_prob_by_target[target.name] = float(parameters.get("mean_best_prob", "nan"))
            norm_info_gain_by_target[target.name] = float(parameters.get("norm_info_gain", "nan"))
            localized_site_count += expected
            best_probability_sum += sum(ordered[:expected])
        if not localized_site_count:
            raise RuntimeError(
                f"adapter scan {adapter_scan} has no targeted modification to localize"
            )

        modification_info = hit.find("p:modification_info", namespace)
        assigned_by_target = {target.name: [] for target in targets}
        output_untargeted = []
        if modification_info is not None:
            if "mod_nterm_mass" in modification_info.attrib:
                nterm_delta = float(modification_info.attrib["mod_nterm_mass"]) - NTERM_H
                terminal_target = match_target("n", nterm_delta, targets)
                if terminal_target is None:
                    output_untargeted.append(f"n_term:{nterm_delta:.6f}")
                else:
                    assigned_by_target[terminal_target.name].append(0)
            for modification in modification_info.findall("p:mod_aminoacid_mass", namespace):
                position = int(modification.attrib["position"])
                residue = peptide[position - 1]
                delta_mass = float(modification.attrib["mass"]) - RESIDUE_MASS[residue]
                target = match_target(residue, delta_mass, targets)
                if target is None:
                    output_untargeted.append(f"{position}:{delta_mass:.6f}")
                else:
                    assigned_by_target[target.name].append(position)
        assigned_by_target = {
            name: sorted(positions) for name, positions in assigned_by_target.items()
        }
        for target in targets:
            if len(assigned_by_target[target.name]) != len(initial_target_sites.get(target.name, ())):
                raise RuntimeError(f"PTMProphet changed {target.name} count at scan {adapter_scan}")
        if peptide != mapping_row["initial_backbone"]:
            raise RuntimeError(f"PTMProphet changed peptide backbone at scan {adapter_scan}")
        # The flattened views below concatenate targets in declaration order, so
        # a single-target run keeps exactly the schema the phospho-only sessions read.
        flat_sites = [site for target in targets for site in sites_by_target[target.name]]
        rows.append(
            {
                "adapter_scan": adapter_scan,
                "spectrum_index": int(mapping_row["spectrum_index"]),
                "spectrum_id": mapping_row["spectrum_id"],
                "model": mapping_row["model"],
                "peptide_backbone": peptide,
                "initial_proforma": mapping_row["initial_proforma"],
                "initial_target_sites": initial_target_sites,
                "assigned_target_sites": assigned_by_target,
                "target_counts": {
                    target.name: len(initial_target_sites.get(target.name, ()))
                    for target in targets
                },
                "site_probabilities_by_target": {
                    name: [site["probability"] for site in sites]
                    for name, sites in sites_by_target.items()
                },
                "site_positions_by_target": {
                    name: [site["position"] for site in sites]
                    for name, sites in sites_by_target.items()
                },
                "mbpr_by_target": mbpr_by_target,
                "ptm_peptide_by_target": ptm_peptide_by_target,
                "mean_best_prob_reported_by_target": mean_best_prob_by_target,
                "norm_info_gain_by_target": norm_info_gain_by_target,
                "auxiliary_ptmprophet_results": auxiliary_results,
                "localized_site_count": localized_site_count,
                "expected_site_errors": localized_site_count - best_probability_sum,
                "initial_phospho_sites": mapping_row["initial_phospho_sites"],
                "assigned_phospho_sites": assigned_by_target.get("Phospho", []),
                "phospho_count": len(mapping_row["initial_phospho_sites"]),
                "site_positions": [site["position"] for site in flat_sites],
                "site_residues": [site["residue"] for site in flat_sites],
                "site_probabilities": [site["probability"] for site in flat_sites],
                "site_probability_sum": sum(site["probability"] for site in flat_sites),
                "mbpr": best_probability_sum / localized_site_count,
                "mean_best_prob_reported": mean_best_prob_by_target[targets[0].name],
                "norm_info_gain": norm_info_gain_by_target[targets[0].name],
                "ptm_peptide": ptm_peptide_by_target[targets[0].name],
                "initial_nonphospho_modifications": mapping_row["initial_nonphospho_modifications"],
                "output_nonphospho_modifications": sorted(output_untargeted),
            }
        )
    if len(rows) != len(mapping):
        raise RuntimeError(f"parsed {len(rows)} PTMProphet rows for {len(mapping)} mapped PSMs")
    return sorted(rows, key=lambda row: row["adapter_scan"])
