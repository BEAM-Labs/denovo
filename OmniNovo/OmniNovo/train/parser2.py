"""Mass spectrometry data parsers for MGF, mzML, and mzXML formats."""
from loguru import logger
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from pyteomics.mgf import MGF


class BaseParser(ABC):
    """A base parser class for mass spectrometry data files.

    Parameters
    ----------
    ms_data_file : str or Path
        The mass spectrometry data file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If None,
        any precursor charge is accepted.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(self, ms_data_file, ms_level, valid_charge=None, id_type="scan"):
        """Initialize the BaseParser."""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.id_type = id_type
        self.offset = None
        self.precursor_mz = []
        self.precursor_charge = []
        self.scan_id = []
        self.mz_arrays = []
        self.intensity_arrays = []

    @abstractmethod
    def open(self):
        """Open the file as an iterable."""
        pass

    @abstractmethod
    def parse_spectrum(self, spectrum):
        """Parse a single spectrum."""
        pass

    def read(self):
        """Read the mass spectrometry data file."""
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    self.parse_spectrum(spectrum)
                except (IndexError, KeyError, ValueError):
                    n_skipped += 1

        if n_skipped:
            logger.warning(f"Skipped {n_skipped} spectra with invalid precursor info")

        self.scan_id = np.array(self.scan_id)

        # Build the index
        sizes = np.array([0] + [len(s) for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()

    @property
    def n_spectra(self):
        """The number of spectra."""
        return self.offset.shape[0]

    @property
    def n_peaks(self):
        """The total number of peaks in the file."""
        mz_arrays_temp = np.concatenate(self.mz_arrays)
        return mz_arrays_temp.shape[0]


class MzmlParser(BaseParser):
    """Parse mass spectra from an mzML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None):
        """Initialize the MzmlParser."""
        super().__init__(ms_data_file, ms_level=ms_level, valid_charge=valid_charge)

    def open(self):
        """Open the mzML file for reading."""
        return MzML(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum from mzML format."""
        if spectrum["ms level"] != self.ms_level:
            return

        if self.ms_level > 1:
            precursor = spectrum["precursorList"]["precursor"][0]
            precursor_ion = precursor["selectedIonList"]["selectedIon"][0]
            precursor_mz = float(precursor_ion["selected ion m/z"])
            if "charge state" in precursor_ion:
                precursor_charge = int(precursor_ion["charge state"])
            elif "possible charge state" in precursor_ion:
                precursor_charge = int(precursor_ion["possible charge state"])
            else:
                precursor_charge = 0
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(list(spectrum["m/z array"]))
            self.intensity_arrays.append(list(spectrum["intensity array"]))
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum["id"]))


class MzxmlParser(BaseParser):
    """Parse mass spectra from an mzXML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzXML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None):
        """Initialize the MzxmlParser."""
        super().__init__(ms_data_file, ms_level=ms_level, valid_charge=valid_charge)

    def open(self):
        """Open the mzXML file for reading."""
        return MzXML(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum from mzXML format."""
        if spectrum["msLevel"] != self.ms_level:
            return

        if self.ms_level > 1:
            precursor = spectrum["precursorMz"][0]
            precursor_mz = float(precursor["precursorMz"])
            precursor_charge = int(precursor.get("precursorCharge", 0))
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(list(spectrum["m/z array"]))
            self.intensity_arrays.append(list(spectrum["intensity array"]))
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum["id"]))


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    ms_data_file : str or Path
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges.
    annotationsLabel : bool
        If True, read peptide annotations from the 'seq' field;
        otherwise use the 'title' field as the identifier.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None, annotationsLabel=False):
        """Initialize the MgfParser."""
        super().__init__(ms_data_file, ms_level=ms_level, valid_charge=valid_charge, id_type="index")
        self.annotationsLabel = annotationsLabel
        self.annotations = []
        self.titles = []
        self.retention_times = []
        self.precursor_intensities = []
        self._counter = 0

    def open(self):
        """Open the MGF file for reading."""
        return MGF(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum from MGF format."""
        if self.ms_level > 1:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            pep = spectrum["params"].get("seq", "") or spectrum["params"].get("pep", "") or ""
            self.annotations.append(pep)

            self.titles.append(spectrum["params"].get("title", ""))

            rt = spectrum["params"].get("rtinseconds", -1.0)
            try:
                rt = float(rt)
            except (ValueError, TypeError):
                rt = -1.0
            self.retention_times.append(rt)

            pepmass = spectrum["params"].get("pepmass", (0,))
            precursor_intensity = float(pepmass[1]) if len(pepmass) > 1 else -1.0
            self.precursor_intensities.append(precursor_intensity)

            self.mz_arrays.append(list(spectrum["m/z array"]))
            self.intensity_arrays.append(list(spectrum["intensity array"]))
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(self._counter)

        self._counter += 1


def _parse_scan_id(scan_str):
    """Remove the string prefix from the scan ID.

    Parameters
    ----------
    scan_str : str
        The scan ID string.

    Returns
    -------
    int
        The scan ID number.
    """
    try:
        return int(scan_str)
    except ValueError:
        try:
            return int(scan_str[scan_str.find("scan=") + len("scan="):])
        except ValueError:
            pass

    raise ValueError("Failed to parse scan number")
