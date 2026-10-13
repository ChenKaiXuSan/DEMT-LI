"""Build ``patient_XX.npz`` model inputs from POPI DICOM, surrogate PNGs and the RPM CSV.

Each file holds ``target`` (3,128,128,128) HU volumes for the 50/70/90% phases,
``surrogate`` (3,3,128,128) coronal 2D frames, ``rpm`` (3,) 1D signal and ``patient``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import popi
from .preprocessing import load_ct_phase, load_png_sequence, read_rpm
from .protocol import PATIENTS, PHASES, folds, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--popi-root", type=Path, required=True,
                        help="directory created by `python -m project.dataloader.popi --output ...`")
    parser.add_argument("--png-root", type=Path, required=True,
                        help="contains 2DCT-<patient>/<phase>.png coronal surrogate frames")
    parser.add_argument("--rpm-csv", type=Path, required=True,
                        help="6 x 10 CSV of diaphragm-apex positions (rows: patients, columns: 0%%..90%%)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dicom-cache", type=Path, default=None,
                        help="where DICOM archives are extracted (default: <output>/dicom)")
    args = parser.parse_args()
    cache = args.dicom_cache or args.output / "dicom"
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for patient in PATIENTS:
        destination = args.output / f"patient_{patient:02d}.npz"
        if destination.exists():
            raise FileExistsError(destination)
        target = np.stack([load_ct_phase(popi.phase_dir(args.popi_root, patient, phase, cache))
                           for phase in PHASES])
        png_dir = args.png_root / f"2DCT-{patient}"
        surrogate = load_png_sequence(png_dir)
        rpm = read_rpm(args.rpm_csv, patient)
        if (target.shape != (3, 128, 128, 128) or surrogate.shape != (3, 3, 128, 128)
                or not all(np.isfinite(a).all() for a in (target, surrogate, rpm))):
            raise ValueError(f"invalid inputs for patient {patient}")
        np.savez_compressed(destination, patient=patient, target=target, surrogate=surrogate, rpm=rpm)
        records.append(dict(patient=patient, sha256=sha256(destination),
                            png_sha256={p: sha256(png_dir / f"{p}.png") for p in PHASES},
                            dicom_archive_sha256=sha256(popi.dicom_archive(args.popi_root, patient)),
                            rpm=rpm.tolist()))
        print(f"patient {patient}: written", flush=True)
    save_json(args.output / "manifest.json", dict(
        phases=list(PHASES), folds=folds(), patients=records, rpm_csv_sha256=sha256(args.rpm_csv),
        ct="POPI DICOM slices 10-137; crop 320x320 around the slice centre; torchvision bilinear "
           "antialiased resize to 128x128; Hounsfield units",
        surrogate="PNG -> RGB -> ToTensor -> Resize(128) -> Normalize(0.45, 0.225) -> /255",
        rpm="diaphragm-apex position at the 50/70/90% phases"))


if __name__ == "__main__":
    main()
