"""Input preprocessing: POPI 4D-CT DICOM, coronal 2D surrogate PNGs and the 1D RPM signal.

The CT path reproduces the original data loader: per axial DICOM slice, crop rows
[h/2-190, h/2+130) and columns [w/2-160, w/2+160) (320 x 320), resize to 128 x 128
with torchvision bilinear interpolation (antialias on), skip the first 10 slices and
keep the next 128. Hounsfield units are retained; no intensity normalization is applied.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import resize

from .protocol import PHASES

SIZE = 128
SKIP_SLICES = 10
CROP_ROWS = (-190, 130)
CROP_COLUMNS = (-160, 160)
RPM_COLUMNS = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9}


def preprocess_ct_slice(image):
    """(1,H,W) float32 tensor in HU -> (1,128,128)."""
    h, w = image.shape[1] // 2, image.shape[2] // 2
    cropped = image[:, h + CROP_ROWS[0]:h + CROP_ROWS[1], w + CROP_COLUMNS[0]:w + CROP_COLUMNS[1]]
    return resize(cropped, size=[SIZE, SIZE], antialias=True)


def load_ct_phase(phase_dir):
    """Read one phase directory of 2D DICOM slices and return a (128,128,128) HU volume."""
    import SimpleITK as sitk
    files = sorted(p for p in Path(phase_dir).iterdir() if p.suffix.lower() == ".dcm")
    selected = files[SKIP_SLICES:SKIP_SLICES + SIZE]
    if len(selected) != SIZE:
        raise ValueError(f"expected at least {SKIP_SLICES + SIZE} DICOM slices in {phase_dir}")
    slices = []
    for path in selected:
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        tensor = torch.from_numpy(array).to(torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None]
        slices.append(preprocess_ct_slice(tensor)[0].numpy())
    return np.stack(slices).astype(np.float32)


SURROGATE_TRANSFORM = Compose([ToTensor(), Resize([SIZE, SIZE]), Normalize(.45, .225), lambda x: x / 255.])


def load_png_sequence(directory, phases=PHASES):
    """Coronal surrogate frames ``<phase>.png`` -> (C=3, T, 128, 128) float32 (original transform)."""
    frames = []
    for phase in phases:
        with Image.open(Path(directory) / f"{phase}.png") as image:
            frames.append(SURROGATE_TRANSFORM(image.convert("RGB")))
    return torch.stack(frames, dim=1).numpy().astype(np.float32)


def read_rpm(csv_path, patient, phases=PHASES):
    """1D diaphragm-apex signal: CSV with one row per patient and ten phase columns (0%..90%)."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    if len(rows) < patient or len(rows[patient - 1]) != 10:
        raise ValueError("expected one row of ten phase values per patient")
    return np.array([float(rows[patient - 1][RPM_COLUMNS[p]]) for p in phases], dtype=np.float32)
