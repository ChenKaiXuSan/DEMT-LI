"""Experiment 2: fixed-model sensitivity to regional 2D surrogate inputs (Figs. 12-14).

For each held-out patient the frozen DEMT-LI checkpoint of that fold is reused
without retraining. Each coronal frame is min-max rescaled to [0, 1] over the whole
frame, one region (half, quadrant or 4x4 cell) is cropped and resized back to
128 x 128 (bilinear, antialiased), and the same region is used for all phases. The 3D
input follows the test protocol (Phase 1 CT only).
Scores are computed on whole reconstructed volumes. This is an input-sensitivity
analysis, not an attention or attribution method.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import interpolate

from ..utils.metrics import METRICS, image_metrics
from ..models.network import build_model, deterministic, load_checkpoint, predict, tensors
from ..dataloader.protocol import PATIENTS, PHASES, load_cases

CONDITIONS = (["full", "left", "right", "upper", "lower"] + [f"grid2_{i}" for i in range(1, 5)]
              + [f"grid4_{i}" for i in range(1, 17)])


def regional_input(sequence, condition, size=128):
    """(C=3, T=3, H, W) surrogate -> regional input; grid cells are numbered row-major from 1."""
    x = np.array(sequence, dtype=np.float32, copy=True)
    if x.ndim != 4 or x.shape[:2] != (3, 3) or not np.isfinite(x).all():
        raise ValueError("expected finite C=3, T=3, H, W surrogate")
    if condition == "full":
        return x
    h, w = x.shape[-2:]
    low = x.min(axis=(0, 2, 3), keepdims=True)
    span = x.max(axis=(0, 2, 3), keepdims=True) - low
    x = (x - low) / np.where(span > 0, span, 1)
    halves = {"left": (slice(None), slice(0, w // 2)), "right": (slice(None), slice(w // 2, None)),
              "upper": (slice(0, h // 2), slice(None)), "lower": (slice(h // 2, None), slice(None))}
    if condition in halves:
        rows, columns = halves[condition]
    else:
        prefix, index = condition.split("_")
        n, i = int(prefix.removeprefix("grid")), int(index) - 1
        if prefix not in ("grid2", "grid4") or not 0 <= i < n * n or h % n or w % n:
            raise ValueError(f"invalid region: {condition}")
        row, column = divmod(i, n)
        rows = slice(row * h // n, (row + 1) * h // n)
        columns = slice(column * w // n, (column + 1) * w // n)
    tensor = torch.from_numpy(x[:, :, rows, columns].transpose(1, 0, 2, 3).copy())
    resized = interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return resized.numpy().transpose(1, 0, 2, 3).copy()


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deterministic()
    device = torch.device(args.device)
    rows = []
    for patient in PATIENTS:
        folder = args.run / "image_2d" / f"fold_{patient:02d}"
        checkpoint = load_checkpoint(folder / "best.pt")
        if checkpoint["split"]["test"] != patient:
            raise ValueError("fold/test-patient mismatch")
        model = build_model("image_2d")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval().requires_grad_(False)
        case = load_cases(args.data, [patient])[patient]
        ct, _, rpm, _ = tensors(case, device, "image_2d")  # test protocol: Phase 1 CT only
        with torch.inference_mode():
            for condition in CONDITIONS:
                surrogate = torch.from_numpy(regional_input(case["surrogate"], condition))[None].to(device)
                prediction = predict(model, "image_2d", ct, surrogate, rpm)[0][0, 0].cpu().numpy()
                for j, phase in enumerate(PHASES):
                    rows.append(dict(patient=patient, phase=phase, condition=condition,
                                     **image_metrics(case["target"][j], prediction[j])))
                print(f"patient {patient}: {condition}", flush=True)
        del model, ct, rpm
        torch.cuda.empty_cache()
    write_csv(args.output / "per_phase.csv", rows)
    people = []
    for patient in PATIENTS:
        for condition in CONDITIONS:
            selected = [r for r in rows if r["patient"] == patient and r["condition"] == condition]
            people.append(dict(patient=patient, condition=condition,
                               **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}))
    write_csv(args.output / "per_patient.csv", people)


if __name__ == "__main__":
    main()
