"""Secondary geometric analyses of the saved test outputs (Tables V/VI, Figs. 20-22).

* DVF plausibility on the saved 128^3 grid: non-positive Jacobian fraction,
  mean |J - 1| and displacement magnitude (one-voxel border excluded).
* Anatomical-landmark TRE (POPI point correspondences; cases 1-3 have 70/90%
  landmarks, all cases have 50%). Landmarks are mapped into the model grid through
  the same crop/resize as the CT; the displacement is applied as the network's
  spatial transformer samples it, S(q) = (q + u(q)) * 128/127 - 0.5 (align_corners=False).
  Millimetres use the native DICOM spacing scaled by the 2.5x in-plane resize.
* Resolution sensitivity (DEMT-LI only): native 320 x 320 crop vs. (a) its
  down/up-sampled copy and (b) the prediction returned to the native crop;
  mean axial slice-wise SSIM, PSNR (4000 HU) and MAE.
Statistics: phases averaged within patient, sample SD, exact sign-flip, bootstrap CI, Holm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity
from skimage.transform import resize

from ..dataloader import popi
from ..utils.metrics import paired_statistics
from ..dataloader.preprocessing import CROP_COLUMNS, CROP_ROWS, SIZE, SKIP_SLICES
from ..dataloader.protocol import METHODS, PHASES, folds
from .summarize import load_prediction

DVF_METRICS = ("folding_fraction", "jacobian_p01", "jacobian_median", "jacobian_p99",
               "mean_abs_jacobian_change", "displacement_mean_vox", "displacement_p95_vox",
               "displacement_max_vox", "gradient_energy")
DVF_TESTED = ["folding_fraction", "mean_abs_jacobian_change", "displacement_mean_vox", "gradient_energy"]
TRE_METRICS = ["baseline_tre_mean_mm", "sampler_tre_mean_mm", "raw_tre_mean_mm"]


# --------------------------------------------------------------------------- DVF
def flow_xyz(saved, time_index):
    """Saved (C=3, T=3, Z, Y, X) flow with channels (z, y, x) -> (Z, Y, X, 3) in (x, y, z) order."""
    field = np.moveaxis(np.asarray(saved)[[2, 1, 0], time_index], 0, -1)
    if not np.isfinite(field).all():
        raise ValueError("nonfinite flow")
    return field


def jacobian_determinant(dvf_xyz):
    dvf_xyz = np.asarray(dvf_xyz, dtype=np.float64)
    ux, uy, uz = (dvf_xyz[..., c] for c in range(3))
    dux_dz, dux_dy, dux_dx = np.gradient(ux, edge_order=2)
    duy_dz, duy_dy, duy_dx = np.gradient(uy, edge_order=2)
    duz_dz, duz_dy, duz_dx = np.gradient(uz, edge_order=2)
    jacobian = np.empty(dvf_xyz.shape[:-1] + (3, 3), dtype=np.float64)
    jacobian[..., 0, :] = np.stack([1. + dux_dx, dux_dy, dux_dz], axis=-1)
    jacobian[..., 1, :] = np.stack([duy_dx, 1. + duy_dy, duy_dz], axis=-1)
    jacobian[..., 2, :] = np.stack([duz_dx, duz_dy, 1. + duz_dz], axis=-1)
    return np.linalg.det(jacobian)


def dvf_metrics(dvf_xyz, border=1):
    dvf_xyz = np.asarray(dvf_xyz, dtype=np.float64)
    interior = (slice(border, -border),) * 3
    determinant = jacobian_determinant(dvf_xyz)[interior]
    displacement = np.linalg.norm(dvf_xyz[interior], axis=-1)
    gradients = [g for c in range(3) for g in np.gradient(dvf_xyz[..., c], edge_order=2)]
    energy = np.sum(np.stack(gradients, axis=-1) ** 2, axis=-1)[interior]
    return dict(folding_fraction=float(np.mean(determinant <= 0.)),
                jacobian_p01=float(np.quantile(determinant, .01)),
                jacobian_median=float(np.median(determinant)),
                jacobian_p99=float(np.quantile(determinant, .99)),
                mean_abs_jacobian_change=float(np.mean(np.abs(determinant - 1.))),
                displacement_mean_vox=float(np.mean(displacement)),
                displacement_p95_vox=float(np.quantile(displacement, .95)),
                displacement_max_vox=float(np.max(displacement)),
                gradient_energy=float(np.mean(energy)))


# --------------------------------------------------------------------------- TRE
def read_geometry(phase_dir):
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in sorted(Path(phase_dir).glob("*.dcm"))])
    return reader.Execute()


def physical_to_model_xyz(points, image):
    """Physical landmark coordinates -> continuous (x, y, z) indices on the 128^3 model grid."""
    index = np.asarray([image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in points])
    width, height = image.GetSize()[:2]
    x0, y0 = width // 2 + CROP_COLUMNS[0], height // 2 + CROP_ROWS[0]
    span_x, span_y = CROP_COLUMNS[1] - CROP_COLUMNS[0], CROP_ROWS[1] - CROP_ROWS[0]
    out = np.empty_like(index)
    out[:, 0] = (index[:, 0] - x0 + .5) * SIZE / span_x - .5
    out[:, 1] = (index[:, 1] - y0 + .5) * SIZE / span_y - .5
    out[:, 2] = index[:, 2] - SKIP_SLICES
    return out


def tre_values(source, target, field, spacing):
    """Target landmarks q are sampled through the DVF and compared with source landmarks p (mm)."""
    p, q = np.asarray(source), np.asarray(target)
    shape = np.asarray(field.shape[:3][::-1], dtype=float)
    u = np.stack([map_coordinates(field[..., i], q[:, ::-1].T, order=1, mode="nearest", prefilter=False)
                  for i in range(3)], axis=1)
    raw = q + u
    mapped = raw * (shape / (shape - 1)) - .5
    return dict(baseline_mm=np.linalg.norm((q - p) * spacing, axis=1),
                raw_mm=np.linalg.norm((raw - p) * spacing, axis=1),
                sampler_mm=np.linalg.norm((mapped - p) * spacing, axis=1),
                mapped_outside=np.any((mapped < 0) | (mapped > shape - 1), axis=1))


def landmark_geometry(popi_root, cache):
    """Return {(patient, phase): (source_xyz, target_xyz, spacing_xyz, n_in_crop, n_total)}."""
    geometry = {}
    for patient in popi.CASES:
        source_image = read_geometry(popi.phase_dir(popi_root, patient, 50, cache))
        spacing = np.asarray(source_image.GetSpacing()) * [320 / SIZE, 320 / SIZE, 1.]
        source = physical_to_model_xyz(np.atleast_2d(np.loadtxt(popi.landmark_file(popi_root, patient, 50))),
                                       source_image)
        for phase in (PHASES if 70 in popi.LANDMARK_PHASES[patient] else (50,)):
            image = read_geometry(popi.phase_dir(popi_root, patient, phase, cache))
            target = physical_to_model_xyz(
                np.atleast_2d(np.loadtxt(popi.landmark_file(popi_root, patient, phase))), image)
            inside = np.all((source >= 0) & (source <= SIZE - 1) & (target >= 0) & (target <= SIZE - 1), axis=1)
            geometry[patient, phase] = (source[inside], target[inside], spacing, int(inside.sum()), len(inside))
    return geometry


# --------------------------------------------------------------------------- resolution
def load_native_crop(phase_dir):
    import SimpleITK as sitk
    files = sorted(Path(phase_dir).glob("*.dcm"))[SKIP_SLICES:SKIP_SLICES + SIZE]
    slices = []
    for path in files:
        image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
        h, w = image.shape[0] // 2, image.shape[1] // 2
        slices.append(image[h + CROP_ROWS[0]:h + CROP_ROWS[1], w + CROP_COLUMNS[0]:w + CROP_COLUMNS[1]])
    return np.stack(slices).astype(np.float32)


def _resize(volume, shape, downsampling):
    return resize(volume, shape, order=1, mode="reflect", anti_aliasing=downsampling,
                  preserve_range=True).astype(np.float32)


def _slice_metrics(reference, prediction):
    error = reference.astype(np.float64) - prediction.astype(np.float64)
    mse = float(np.mean(error ** 2))
    data_range = float(reference.max() - reference.min())
    ssim = np.mean([structural_similarity(r, p, data_range=data_range, gaussian_weights=True, sigma=1.5,
                                          use_sample_covariance=False)
                    for r, p in zip(reference, prediction)])
    return dict(ssim=float(ssim), psnr_db=float(20 * np.log10(4000. / np.sqrt(mse))),
                mae_hu=float(np.mean(np.abs(error))))


def resolution_metrics(native, prediction):
    roundtrip = _resize(_resize(native, (SIZE,) * 3, True), native.shape, False)
    returned = _resize(np.asarray(prediction, dtype=np.float32), native.shape, False)
    return {**{f"ceiling_{k}": v for k, v in _slice_metrics(native, roundtrip).items()},
            **{f"prediction_{k}": v for k, v in _slice_metrics(native, returned).items()}}


# --------------------------------------------------------------------------- driver
def summarize(frame, metrics):
    people = frame.groupby(["method", "patient"], as_index=False)[metrics].mean()
    rows = []
    for method, group in people.groupby("method"):
        row = dict(method=method, n_patients=len(group))
        for metric in metrics:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return people, pd.DataFrame(rows)


def compare(people, metrics):
    return pd.DataFrame(paired_statistics(people.to_dict("records"), metrics=metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--popi-root", type=Path, required=True)
    parser.add_argument("--dicom-cache", type=Path, required=True,
                        help="extracted DICOM directory (e.g. <data>/dicom from prepare_data)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    geometry = landmark_geometry(args.popi_root, args.dicom_cache)
    dvf_rows, tre_rows, resolution_rows = [], [], []
    for method in METHODS:
        for split in folds():
            patient = split["test"]
            prediction, saved_flow = load_prediction(args.run, method, split)
            for index, phase in enumerate(PHASES):
                field = flow_xyz(saved_flow, index)
                dvf_rows.append(dict(method=method, patient=patient, phase=phase, **dvf_metrics(field)))
                if (patient, phase) in geometry:
                    source, target, spacing, n_in, n_total = geometry[patient, phase]
                    values = tre_values(source, target, field, spacing)
                    row = dict(method=method, patient=patient, source_phase=50, target_phase=phase,
                               analysis_type="identity_50_to_50" if phase == 50 else "motion_50_to_target",
                               n_landmarks_total=n_total, n_landmarks_in_crop=n_in,
                               mapped_outside_count=int(values["mapped_outside"].sum()))
                    for label in ("baseline", "sampler", "raw"):
                        row[f"{label}_tre_mean_mm"] = float(values[f"{label}_mm"].mean())
                        row[f"{label}_tre_median_mm"] = float(np.median(values[f"{label}_mm"]))
                        row[f"{label}_tre_p95_mm"] = float(np.quantile(values[f"{label}_mm"], .95))
                    tre_rows.append(row)
                if method == "image_2d":
                    native = load_native_crop(popi.phase_dir(args.popi_root, patient, phase, args.dicom_cache))
                    resolution_rows.append(dict(method=method, patient=patient, phase=phase,
                                                **resolution_metrics(native, prediction[index])))
                print(f"{method} patient {patient} phase {phase}", flush=True)
    dvf, tre, res = (pd.DataFrame(rows) for rows in (dvf_rows, tre_rows, resolution_rows))
    dvf.to_csv(args.output / "dvf_per_phase.csv", index=False)
    people, summary = summarize(dvf, list(DVF_METRICS))
    people.to_csv(args.output / "dvf_per_patient.csv", index=False)
    summary.to_csv(args.output / "dvf_summary.csv", index=False)
    compare(people, DVF_TESTED).to_csv(args.output / "dvf_paired_statistics.csv", index=False)
    tre.to_csv(args.output / "landmark_tre.csv", index=False)
    motion = tre[tre.analysis_type == "motion_50_to_target"]
    people, summary = summarize(motion, TRE_METRICS)
    people.to_csv(args.output / "tre_motion_per_patient.csv", index=False)
    summary.to_csv(args.output / "tre_motion_summary.csv", index=False)
    compare(people, ["sampler_tre_mean_mm"]).to_csv(args.output / "tre_paired_statistics.csv", index=False)
    by_phase = []
    for phase, frame in tre.groupby("target_phase"):
        by_phase.append(summarize(frame, TRE_METRICS)[1].assign(target_phase=phase))
    pd.concat(by_phase, ignore_index=True).to_csv(args.output / "tre_by_phase_summary.csv", index=False)
    res.to_csv(args.output / "resolution_per_phase.csv", index=False)
    columns = [c for c in res.columns if c.startswith(("ceiling_", "prediction_"))]
    pd.DataFrame([dict(metric=c, n_volumes=len(res), mean=res[c].mean(), sd=res[c].std(ddof=1))
                  for c in columns]).to_csv(args.output / "resolution_volume_summary.csv", index=False)


if __name__ == "__main__":
    main()
