#!/usr/bin/env python3
"""Generate deterministic peak-replacement spectrum decoys in LMDB and MGF."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import lmdb
import numpy as np
import polars as pl


ALGORITHM_VERSION = "pair_sampling_no_target_mz_collision_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(data.dtype.str.encode())
    digest.update(np.asarray(data.shape, dtype=np.int64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def mz_keys(mz: np.ndarray) -> np.ndarray:
    """Frozen m/z collision rule: float32 m/z rounded to the nearest 1e-5 bin."""
    return np.rint(np.asarray(mz, np.float32).astype(np.float64) * 100000).astype(np.int64)


def read_targets(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False, readahead=False)
    spectra: list[dict] = []
    pool_mz: list[np.ndarray] = []
    pool_intensity: list[np.ndarray] = []
    with env.begin() as txn:
        n = int(txn.get(b"n_spectra"))
        for index in range(n):
            value = txn.get(str(index).encode())
            if value is None:
                raise RuntimeError(f"missing target LMDB key {index}")
            record = pickle.loads(value)
            mz = np.asarray(record["mz_array"], np.float32)
            intensity = np.asarray(record["intensity_array"], np.float32)
            if (
                len(mz) == 0
                or len(mz) != len(intensity)
                or not np.all(np.isfinite(mz))
                or not np.all(np.isfinite(intensity))
                or np.any(mz <= 0)
                or np.any(intensity <= 0)
            ):
                raise RuntimeError(f"invalid target peaks at index {index}")
            if len(np.unique(mz_keys(mz))) != len(mz):
                raise RuntimeError(f"target has duplicate five-decimal m/z at index {index}")
            spectra.append(
                {
                    "mz": mz,
                    "intensity": intensity,
                    "precursor_mz": float(record["precursor_mz"]),
                    "precursor_charge": int(record["precursor_charge"]),
                    "title": str(record.get("title") or f"index={index}"),
                }
            )
            pool_mz.append(mz)
            pool_intensity.append(intensity)
    env.close()
    return spectra, np.concatenate(pool_mz), np.concatenate(pool_intensity)


def sample_noise_indices(
    rng: np.random.Generator,
    pool_keys: np.ndarray,
    forbidden_keys: set[int],
    count: int,
) -> tuple[np.ndarray, int]:
    """Sample complete peak-pair indices, rejecting m/z collisions deterministically."""
    accepted: list[int] = []
    rejected = 0
    while len(accepted) < count:
        remaining = count - len(accepted)
        candidates = rng.integers(0, len(pool_keys), size=remaining, dtype=np.int64)
        for pool_index in candidates:
            key = int(pool_keys[int(pool_index)])
            if key in forbidden_keys:
                rejected += 1
                continue
            forbidden_keys.add(key)
            accepted.append(int(pool_index))
    return np.asarray(accepted, np.int64), rejected


def generate_rate(
    dataset: str,
    target_path: Path,
    spectra: list[dict],
    pool_mz: np.ndarray,
    pool_intensity: np.ndarray,
    pool_keys: np.ndarray,
    target_sha256: str,
    pool_pair_hash: str,
    output_root: Path,
    seed_label: str,
    seed_integer: int,
    rho: float,
    n_generate: int,
) -> dict:
    rate_name = f"rho_{rho:.2f}"
    final_dir = output_root / dataset / f"seed_{seed_label}" / rate_name
    partial_dir = final_dir.with_name(final_dir.name + f".partial.{os.getpid()}")
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed output: {final_dir}")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    lmdb_path = partial_dir / "decoy.lmdb"
    mgf_path = partial_dir / "decoy.mgf"
    trace_path = partial_dir / "generation_trace.parquet"
    metadata_path = partial_dir / "generation.json"
    rng = np.random.Generator(np.random.PCG64(seed_integer))
    trace: list[dict] = []
    rejection_total = 0
    output_env = lmdb.open(
        str(lmdb_path),
        map_size=max(2_000_000_000, target_path.stat().st_size * 4),
        subdir=False,
        readonly=False,
        lock=False,
    )
    try:
        with output_env.begin(write=True) as txn, mgf_path.open("w", encoding="utf-8", newline="\n") as mgf:
            for index, target in enumerate(spectra[:n_generate]):
                target_mz = target["mz"]
                target_intensity = target["intensity"]
                n_peaks = len(target_mz)
                n_keep = math.floor(n_peaks * (1 - rho))
                keep_indices = np.sort(
                    rng.choice(n_peaks, size=n_keep, replace=False).astype(np.int64)
                    if n_keep
                    else np.empty(0, np.int64)
                )
                removed_indices = np.setdiff1d(
                    np.arange(n_peaks, dtype=np.int64), keep_indices, assume_unique=True
                )
                forbidden = set(int(value) for value in mz_keys(target_mz))
                inserted_indices, rejected = sample_noise_indices(
                    rng, pool_keys, forbidden, n_peaks - n_keep
                )
                rejection_total += rejected

                output_mz = np.concatenate((target_mz[keep_indices], pool_mz[inserted_indices])).astype(np.float32)
                output_intensity = np.concatenate(
                    (target_intensity[keep_indices], pool_intensity[inserted_indices])
                ).astype(np.float32)
                order = np.argsort(output_mz, kind="stable")
                output_mz = output_mz[order]
                output_intensity = output_intensity[order]
                if len(output_mz) != n_peaks or len(np.unique(mz_keys(output_mz))) != n_peaks:
                    raise RuntimeError(f"postcondition failed at spectrum {index}")

                title = f"seed{seed_label}|r{rho:.2f}|{target['title']}"
                record = {
                    "precursor_mz": target["precursor_mz"],
                    "precursor_charge": target["precursor_charge"],
                    "mz_array": output_mz,
                    "intensity_array": output_intensity,
                    "pep": "",
                    "title": title,
                    "source_index": index,
                    "source_title": target["title"],
                    "decoy_seed": seed_label,
                    "rho": rho,
                }
                txn.put(str(index).encode(), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                mgf.write("BEGIN IONS\n")
                mgf.write(f"TITLE={title}\n")
                mgf.write(f"PEPMASS={target['precursor_mz']}\n")
                mgf.write(f"CHARGE={target['precursor_charge']}+\n")
                mgf.write(f"SCANS={index}\n")
                for mz, intensity in zip(output_mz, output_intensity):
                    mgf.write(f"{float(mz):.5f} {float(intensity):.5f}\n")
                mgf.write("END IONS\n\n")

                output_pairs = np.column_stack((output_mz, output_intensity)).astype(np.float32)
                trace.append(
                    {
                        "dataset": dataset,
                        "seed_label": seed_label,
                        "rng_integer": seed_integer,
                        "rho": rho,
                        "spectrum_index": index,
                        "target_title": target["title"],
                        "n_peaks": n_peaks,
                        "n_keep": n_keep,
                        "n_replace": n_peaks - n_keep,
                        "rejected_noise_draws": rejected,
                        "kept_index_hash_blake2b128": array_hash(keep_indices),
                        "removed_index_hash_blake2b128": array_hash(removed_indices),
                        "inserted_pool_index_hash_blake2b128": array_hash(inserted_indices),
                        "output_pair_hash_blake2b128": array_hash(output_pairs),
                    }
                )
            txn.put(b"n_spectra", str(n_generate).encode())
            txn.put(b"ms_level", b"2")
            txn.put(b"algorithm_version", ALGORITHM_VERSION.encode())
        output_env.sync()
    except Exception:
        output_env.close()
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    output_env.close()

    pl.DataFrame(trace, infer_schema_length=None).write_parquet(trace_path, compression="zstd")
    metadata = {
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": dataset,
        "target_lmdb": str(target_path),
        "target_lmdb_sha256": target_sha256,
        "seed_label": seed_label,
        "rng_integer": seed_integer,
        "bit_generator": "numpy.PCG64 reset independently for each (dataset, seed, rho)",
        "rho": rho,
        "rho_definition": "fraction of original peaks removed and replaced",
        "n_keep_rule": "floor(n_peaks * (1-rho))",
        "noise_pool": "all valid float32 (m/z, intensity) pairs in the target LMDB, in target-index order; no truth labels",
        "noise_pool_peak_count": len(pool_mz),
        "noise_pool_pair_hash_blake2b128": pool_pair_hash,
        "collision_rule": (
            "reject a sampled pair when round(float32_mz*1e5) collides with any m/z in the original "
            "target spectrum or an already accepted inserted pair"
        ),
        "pair_sampling": "complete pool (m/z, intensity) pairs; replacement across spectra; deterministic rejection within spectrum",
        "target_spectra_available": len(spectra),
        "spectra_generated": n_generate,
        "rejected_noise_draws": rejection_total,
        "files": {},
    }
    for path in (lmdb_path, mgf_path, trace_path):
        metadata["files"][path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(partial_dir, final_dir)
    return {"output": str(final_dir), **metadata}
