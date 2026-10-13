"""Image-fidelity metrics, patient-level aggregation and paired statistics."""
from __future__ import annotations

from itertools import product

import numpy as np
from skimage.metrics import structural_similarity

from ..dataloader.protocol import METHODS, PHASES

METRICS = ("ssim", "psnr_db", "mae_hu")
PSNR_RANGE_HU = 4000.
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_RESAMPLES = 10000


def image_metrics(target, prediction):
    """Volume-wide 3D SSIM, PSNR (fixed 4000-HU range) and MAE (HU)."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid prediction")
    error = prediction - target
    mse = np.mean(error ** 2)
    ssim = structural_similarity(target, prediction, data_range=float(np.ptp(target)) or PSNR_RANGE_HU,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return dict(ssim=float(ssim),
                psnr_db=float(20 * np.log10(PSNR_RANGE_HU / np.sqrt(mse))) if mse else float("inf"),
                mae_hu=float(np.mean(np.abs(error))))


def patient_summary(rows, phases=PHASES, metrics=METRICS):
    """Average phases within each patient, then summarize patients (mean, sample SD)."""
    people, aggregate = [], []
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    for method in methods:
        for patient in sorted({r["patient"] for r in rows if r["method"] == method}):
            selected = [r for r in rows if r["method"] == method and r["patient"] == patient
                        and r["phase"] in phases]
            if sorted(r["phase"] for r in selected) != sorted(phases):
                raise ValueError("missing or duplicate phase")
            people.append(dict(method=method, patient=patient,
                               **{k: float(np.mean([r[k] for r in selected])) for k in metrics}))
        selected = [r for r in people if r["method"] == method]
        row = dict(method=method, n_patients=len(selected))
        for key in metrics:
            values = [r[key] for r in selected]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_sd"] = float(np.std(values, ddof=1))
        aggregate.append(row)
    return people, aggregate


def exact_signflip(differences):
    """Exact two-sided paired sign-flip p-value over all 2^n sign assignments."""
    d = np.asarray(differences, dtype=np.float64)
    permutations = [abs(np.mean(d * np.array(sign))) for sign in product([-1, 1], repeat=len(d))]
    return float(np.mean(np.array(permutations) >= abs(d.mean()) - 1e-15))


def bootstrap_ci(differences, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    """Percentile patient-bootstrap 95% CI of the mean paired difference."""
    d = np.asarray(differences, dtype=np.float64)
    index = np.random.default_rng(seed).integers(0, len(d), size=(resamples, len(d)))
    low, high = np.quantile(d[index].mean(axis=1), [.025, .975])
    return float(low), float(high)


def holm(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(np.minimum(1., (len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def paired_statistics(people, metrics=METRICS, proposed=METHODS[0], baselines=METHODS[1:]):
    """DEMT-LI versus each baseline; Holm correction across all rows of the family."""
    rows = []
    for baseline in baselines:
        for metric in metrics:
            a = {r["patient"]: r[metric] for r in people if r["method"] == proposed}
            b = {r["patient"]: r[metric] for r in people if r["method"] == baseline}
            if not a or a.keys() != b.keys():
                raise ValueError("matched patients are required")
            d = np.array([a[p] - b[p] for p in sorted(a)])
            low, high = bootstrap_ci(d)
            rows.append(dict(proposed=proposed, baseline=baseline, metric=metric, n_patients=len(d),
                             mean_difference=float(d.mean()), ci95_low=low, ci95_high=high,
                             p_exact=exact_signflip(d)))
    for row, adjusted in zip(rows, holm([r["p_exact"] for r in rows])):
        row["p_holm"] = float(adjusted)
    return rows
