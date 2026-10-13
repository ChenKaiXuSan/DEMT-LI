"""Primary image-fidelity results (Tables II/III): per-phase metrics, patient means, paired tests.

Phases are averaged within each held-out patient before summarizing (n = 6). The
70/90% summary excludes the reference phase; the unwarped reference (50% volume
copied to 70/90%) is reported as a non-learned comparator.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..utils.metrics import image_metrics, paired_statistics, patient_summary
from ..dataloader.protocol import METHODS, PATIENTS, PHASES, folds


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_prediction(run, method, split):
    folder = Path(run) / method / f"fold_{split['fold']:02d}"
    if json.loads((folder / "status.json").read_text())["state"] != "complete":
        raise ValueError(f"incomplete run: {folder}")
    with np.load(folder / "test_prediction.npz") as saved:
        if int(saved["patient"]) != split["test"]:
            raise ValueError("prediction patient mismatch")
        return saved["prediction"], saved["dvf"]


def load_target(data, patient):
    with np.load(Path(data) / f"patient_{patient:02d}.npz") as case:
        return case["target"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="default: <run>/summary")
    args = parser.parse_args()
    output = args.output or args.run / "summary"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for method in METHODS:
        for split in folds():
            prediction, _ = load_prediction(args.run, method, split)
            target = load_target(args.data, split["test"])
            for i, phase in enumerate(PHASES):
                rows.append(dict(method=method, patient=split["test"], phase=phase,
                                 **image_metrics(target[i], prediction[i])))
    write_csv(output / "per_phase.csv", rows)
    for name, phases in [("all_phases", PHASES), ("motion_phases_70_90", (70, 90))]:
        people, summary = patient_summary(rows, phases)
        write_csv(output / f"{name}_per_patient.csv", people)
        write_csv(output / f"{name}_summary.csv", summary)
        write_csv(output / f"{name}_paired_statistics.csv", paired_statistics(people))
        if name == "all_phases":
            for row in summary:
                print(f"{row['method']:>13}: SSIM {row['ssim_mean']:.4f} ± {row['ssim_sd']:.4f}, "
                      f"PSNR {row['psnr_db_mean']:.2f} ± {row['psnr_db_sd']:.2f} dB, "
                      f"MAE {row['mae_hu_mean']:.2f} ± {row['mae_hu_sd']:.2f} HU")
    reference = []
    for patient in PATIENTS:
        target = load_target(args.data, patient)
        for i in (1, 2):
            reference.append(dict(method="no_surrogate", patient=patient, phase=PHASES[i],
                                  **image_metrics(target[i], target[0])))
    people, summary = patient_summary(reference, (70, 90))
    for row in people + summary:
        row["method"] = "unwarped_reference"
    write_csv(output / "unwarped_reference_motion_phases.csv", people)
    write_csv(output / "unwarped_reference_summary.csv", summary)


if __name__ == "__main__":
    main()
