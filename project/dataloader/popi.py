"""POPI-model 4D-CT cases (Vandemeulebroucke et al.) used in this study."""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://www.creatis.insa-lyon.fr/~srit/POPI/MedPhys11"
CASES = {1: "bl", 2: "ng", 3: "dx", 4: "gt", 5: "mm2", 6: "bh"}
# Cases 1-3 provide landmarks for all ten phases; cases 4-6 only for 0% and 50%.
LANDMARK_PHASES = {p: (tuple(range(0, 100, 10)) if p <= 3 else (0, 50)) for p in CASES}


def case_dir(root, patient):
    return Path(root) / f"patient_{patient:02d}_{CASES[patient]}"


def dicom_archive(root, patient):
    return case_dir(root, patient) / "images" / "4DCT-Dicom.tar.gz"


def landmark_file(root, patient, phase):
    return case_dir(root, patient) / "points" / f"{phase:02d}.pts"


def phase_dir(root, patient, phase, cache):
    """Extract (once) the DICOM archive into ``cache`` and return the phase directory."""
    destination = Path(cache) / f"{patient:02d}"
    marker = destination / ".complete"
    if not marker.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dicom_archive(root, patient), "r:gz") as bundle:
            bundle.extractall(destination, filter="data")
        marker.write_text("extracted\n")
    return destination / f"{phase:02d}"


def download(root):
    for patient, code in CASES.items():
        items = [(f"{BASE_URL}/{code}/dcm/4DCT-Dicom.tar.gz", dicom_archive(root, patient))]
        items += [(f"{BASE_URL}/{code}/pts/{p:02d}.pts", landmark_file(root, patient, p))
                  for p in LANDMARK_PHASES[patient]]
        for url, path in items:
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {url}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Download the six POPI 4D-CT cases and landmarks.")
    parser.add_argument("--output", type=Path, required=True)
    download(parser.parse_args().output)


if __name__ == "__main__":
    main()
