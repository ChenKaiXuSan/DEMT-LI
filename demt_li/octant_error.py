"""Descriptive regional error (Fig. 23): MAE in a fixed 2 x 2 x 2 index-space grid.

Each held-out DEMT-LI prediction (6 patients x 3 phases) is split into eight
64^3 octants of the saved array; octant labels are array-index halves, not anatomy.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from .protocol import PHASES, folds
from .summarize import load_prediction, load_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for split in folds():
        prediction, _ = load_prediction(args.run, "image_2d", split)
        error = np.abs(prediction.astype(np.float64) - load_target(args.data, split["test"]).astype(np.float64))
        for index, phase in enumerate(PHASES):
            values = []
            for z, y, x in product(range(2), repeat=3):
                value = float(error[index, 64 * z:64 * (z + 1), 64 * y:64 * (y + 1), 64 * x:64 * (x + 1)].mean())
                values.append(value)
                rows.append(dict(patient=split["test"], phase=phase, region_id=f"z{z}_y{y}_x{x}",
                                 z_index=z, y_index=y, x_index=x, mae_hu=value))
            np.testing.assert_allclose(np.mean(values), error[index].mean(), rtol=0, atol=1e-10)
    with (args.output / "regional_per_volume.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for z, y, x in product(range(2), repeat=3):
        values = [r["mae_hu"] for r in rows if (r["z_index"], r["y_index"], r["x_index"]) == (z, y, x)]
        summary.append(dict(region_id=f"z{z}_y{y}_x{x}", z_index=z, y_index=y, x_index=x,
                            mean_mae_hu=float(np.mean(values)), sample_sd_mae_hu=float(np.std(values, ddof=1)),
                            n_volumes=len(values)))
    with (args.output / "regional_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['region_id']}: {row['mean_mae_hu']:.1f} HU")


if __name__ == "__main__":
    main()
