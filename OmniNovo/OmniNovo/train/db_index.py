"""LMDB-based spectrum index for efficient data loading."""
import lmdb
import numpy as np
from loguru import logger
from pathlib import Path
from .parser2 import MzmlParser, MzxmlParser, MgfParser
import os
import pickle
import io


class NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that handles numpy 2.0 module renaming."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def listify(obj):
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]
    return list(obj)


class DB_Index:
    """Read, store, and manage a single LMDB file for spectrum data.

    Parameters
    ----------
    db_path : str or Path
        Path to the LMDB database file.
    filenames : list of str, optional
        Mass spectrometry files to index into this database.
    ms_level : int
        The MS level of spectra to store.
    valid_charge : list of int, optional
        Only store spectra with these precursor charges.
    annotated : bool
        Whether the spectra have peptide annotations.
    lock : bool
        If True, open the database in write mode; if False, read-only mode.
    """

    def __init__(self, db_path, filenames=None, ms_level=2, valid_charge=None, annotated=True, lock=True):
        """Initialize the DB_Index."""
        index_path = Path(db_path)
        self.db_path = index_path
        self.lock = lock
        self.filenames = filenames
        self.ms_level = ms_level
        self.valid_charge = valid_charge
        self.annotated = bool(annotated)
        is_exist = index_path.exists()

        self._init_db()

        if not is_exist:
            txn = self.env.begin(write=True)
            txn.put("ms_level".encode(), str(self.ms_level).encode())
            txn.put("n_spectra".encode(), str(0).encode())
            txn.put("n_peaks".encode(), str(0).encode())
            self.n_spectra = 0
            txn.commit()
        else:
            txn = self.env.begin()
            self.n_spectra = int(txn.get("n_spectra".encode()).decode())

        # Close the env for read-only mode; __getitem__ will reopen lazily per-worker.
        # Write paths (lock=True) keep env open since they run in the main process.
        if not self.lock:
            self.env.close()
            self.env = None

        if filenames is not None:
            filenames = listify(filenames)
            logger.info(f"Reading {len(filenames)} files...")
            for ms_file in filenames:
                self.add_file(ms_file)

    def write_to_db(self, parser, n):
        """Write all spectra from the parser to LMDB."""
        txn = self.env.begin(write=True)
        assert n == len(parser.precursor_charge)
        has_titles = hasattr(parser, 'titles') and len(parser.titles) == n
        has_rt = hasattr(parser, 'retention_times') and len(parser.retention_times) == n
        has_intensity = hasattr(parser, 'precursor_intensities') and len(parser.precursor_intensities) == n
        for i in range(len(parser.precursor_charge)):
            collection_data = {
                "precursor_mz": parser.precursor_mz[i],
                "precursor_charge": parser.precursor_charge[i],
                "mz_array": parser.mz_arrays[i],
                "intensity_array": parser.intensity_arrays[i],
                "pep": parser.annotations[i] if parser.annotations[i] else "",
            }
            if has_titles:
                collection_data["title"] = parser.titles[i]
            if has_rt:
                collection_data["retention_time"] = parser.retention_times[i]
            if has_intensity:
                collection_data["precursor_intensity"] = parser.precursor_intensities[i]
            buffers = pickle.dumps(collection_data)
            txn.put(str(self.n_spectra).encode(), buffers)
            self.n_spectra += 1
        txn.put("n_spectra".encode(), str(self.n_spectra).encode())
        txn.commit()

    def __getitem__(self, idx):
        """Retrieve a spectrum by index.

        Returns a tuple of (mz_array, intensity_array, precursor_mz, precursor_charge, peptide, metadata).
        """
        # Lazy-open: each DataLoader worker opens its own LMDB handle on first access.
        if self.env is None:
            self._init_db()
        txn = self.env.begin()
        buffer = txn.get(str(idx).encode())
        data = NumpyCompatUnpickler(io.BytesIO(buffer)).load()
        pep = data["pep"]

        metadata = {
            "title": data.get("title", ""),
            "retention_time": data.get("retention_time", -1.0),
            "precursor_intensity": data.get("precursor_intensity", -1.0),
            "original_pep": data.get("original_pep", ""),
        }

        return (data["mz_array"], data["intensity_array"],
                data["precursor_mz"], data["precursor_charge"], pep, metadata)

    def __len__(self):
        return self.n_spectra

    def add_file(self, a_file):
        """Parse and add a mass spectrometry file to the database."""
        a_file = Path(a_file)
        parser = self._get_parser(a_file)
        parser.read()
        n_spectra_in_file = parser.n_spectra
        self.write_to_db(parser, n_spectra_in_file)

    def _init_db(self):
        """Initialize the LMDB environment."""
        if self.lock:
            self.env = lmdb.open(str(self.db_path), map_size=4099511627776, subdir=False, readonly=False, lock=self.lock)
        else:
            self.env = lmdb.open(str(self.db_path), subdir=False, readonly=True, lock=False)

    def _get_parser(self, ms_data_file):
        """Get the appropriate parser for the given file type."""
        kw_args = dict(ms_level=self.ms_level, valid_charge=self.valid_charge, annotationsLabel=self.annotated)
        if ms_data_file.suffix.lower() == ".mzml":
            return MzmlParser(ms_data_file, **kw_args)
        if ms_data_file.suffix.lower() == ".mzxml":
            return MzxmlParser(ms_data_file, **kw_args)
        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(ms_data_file, **kw_args)
        raise ValueError("Only mzML, mzXML, and MGF files are supported.")
