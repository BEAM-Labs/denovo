#!/usr/bin/env python3
"""Convert a compact adapter MGF to indexed mzML with stable sequential scan IDs."""

from __future__ import annotations

import argparse

from pyopenms import MSExperiment, MascotGenericFile, MzMLFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mgf")
    parser.add_argument("output_mzml")
    args = parser.parse_args()
    experiment = MSExperiment()
    MascotGenericFile().load(args.input_mgf, experiment)
    for index, spectrum in enumerate(experiment):
        spectrum.setNativeID(f"scan={index}")
        spectrum.setMSLevel(2)
        spectrum.setRT(float(index + 1))
    MzMLFile().store(args.output_mzml, experiment)
    print(f"stored {experiment.size()} spectra")


if __name__ == "__main__":
    main()
