"""Model construction, per-method settings, input tensors and inference.

The three networks are loaded from the unmodified original implementation in
``demt_li/legacy`` so that saved checkpoints and numerical behaviour are preserved.

Input convention (important): the 3D encoder receives the complete three-phase CT
sequence (50%, 70%, 90%) during training *and* evaluation, while the spatial
transformer warps only the first (50%) phase. This is therefore not a
reference-volume-only inference protocol.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .protocol import METHODS

LEGACY = Path(__file__).resolve().parent / "legacy"
SOURCES = {
    "image_2d": LEGACY / "image_2d" / "Time_series_seq2seq_4DCT_voxelmorph.py",
    "rpm_1d": LEGACY / "rpm_1d" / "Time_series_seq2seq_4DCT_voxelmorph.py",
    "no_surrogate": LEGACY / "no_surrogate" / "seq2seq_4DCT_voxelmorph.py",
}


def settings(method):
    """Training configuration used for the reported runs."""
    if method not in METHODS:
        raise ValueError(method)
    return dict(lr=.0001 if method == "no_surrogate" else .001, betas=(.9, .999), eps=1e-8,
                weight_decay=0., shuffle=method != "no_surrogate",
                per_axial_slice_flip_probability=.5 if method == "no_surrogate" else 0.,
                epochs=200, batch_size=1, channels=96, seed=42, lambda_smooth=1.,
                scheduler="ReduceLROnPlateau(factor=0.1, patience=10, threshold=1e-4, rel)",
                scheduler_monitor="internal_validation_loss / initial_internal_validation_loss",
                early_stopping=False, input="full_three_phase_CT_sequence",
                warped_volume="first phase (50%) only")


class Recompute(nn.Module):
    """Activation checkpointing around a ConvLSTM cell (memory only; same arithmetic)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def init_hidden(self, *args, **kwargs):
        return self.module.init_hidden(*args, **kwargs)

    def forward(self, input_tensor, cur_state):
        def forward(x, h, c):
            return self.module(x, (h, c))
        if self.training and torch.is_grad_enabled():
            return checkpoint(forward, input_tensor, *cur_state, use_reentrant=False)
        return forward(input_tensor, *cur_state)


def build_model(method, size=128, channels=96, recompute=True):
    common = str(LEGACY / "common")
    sys.path.insert(0, common)
    try:
        spec = importlib.util.spec_from_file_location(f"legacy_{method}", SOURCES[method])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(common)
    model = module.EncoderDecoderConvLSTM(channels, 1, size, size, size)
    if recompute:
        # Checkpoints from the reported runs were saved with these wrappers.
        for name in ("ConvLSTM3d1", "ConvLSTM3d2", "ConvLSTM3d3", "ConvLSTM3d4"):
            setattr(model, name, Recompute(getattr(model, name)))
    return model


def make_optimizer(model, method):
    config = settings(method)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"],
                                 eps=config["eps"], weight_decay=config["weight_decay"])
    return optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


def tensors(case, device, method, training=False):
    """Return (ct, surrogate, rpm, target) tensors for one patient."""
    target = torch.from_numpy(case["target"])[None, None].to(device)
    if training and method == "no_surrogate":
        # The original no-surrogate pipeline applied RandomHorizontalFlip independently to
        # every raw axial slice, including the ten discarded leading slices; reproduced here.
        t, d = target.shape[2:4]
        flips = (torch.rand(t, d + 10)[:, 10:] < .5).to(device)[None, None, :, :, None, None]
        target = torch.where(flips, target.flip(-1), target)
    ct = target.transpose(1, 2).contiguous().clone().detach()
    surrogate = torch.from_numpy(case["surrogate"])[None].to(device)
    rpm = torch.from_numpy(case["rpm"])[None].to(device)
    return ct, surrogate, rpm, target


def patient_order(patients, method):
    if settings(method)["shuffle"]:
        return [patients[i] for i in torch.randperm(len(patients)).tolist()]
    return list(patients)


def predict(model, method, ct, surrogate=None, rpm=None):
    """Return (prediction (B,1,T,D,H,W), DVF (B,3,T,D,H,W)) for a (B,3,1,D,H,W) CT sequence."""
    if ct.ndim != 6 or ct.shape[1:3] != (3, 1):
        raise ValueError("expected a three-phase CT sequence (B,3,1,D,H,W)")
    if method == "image_2d":
        return model(ct, surrogate, future_seq=3)
    if method == "rpm_1d":
        if ct.shape[0] != 1 or rpm.shape != (1, 3):
            raise ValueError("the 1D surrogate requires batch size 1 and three signal values")
        return model(ct, rpm_tensor=rpm[0], future_seq=3)
    if method == "no_surrogate":
        return model(ct, rpm_x=None, rpm_y=None, future_seq=3)
    raise ValueError(method)


def load_checkpoint(path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def deterministic(seed=42, threads=2):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
