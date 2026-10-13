"""Model construction, per-method settings, input tensors and inference.

The three networks in this directory (``Time_series_seq2seq_4DCT_voxelmorph.py``,
``RPM_Time_series_seq2seq_4DCT_voxelmorph.py`` and ``seq2seq_4DCT_voxelmorph.py``,
with ``ConvLSTMCell3d.py``, ``layers.py`` and ``unet_utils.py``) are the original,
unmodified implementation, so saved checkpoints and numerical behaviour are preserved.

CT input protocol (Fig. 6 of the manuscript; ``configs/config.yaml``):
* training, including internal validation for checkpoint selection: the 3D encoder
  receives the time-series CT volumes (50%, 70%, 90%) together with the 2D frames;
* testing: the trained model receives only the Phase 1 (50%) CT volume together with
  the time-series 2D frames.
The predicted DVFs always warp the Phase 1 (50%) CT.
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

from ..dataloader.protocol import METHODS
from ..utils.config import load_config

CT_INPUTS = ("reference", "sequence")  # Phase 1 CT only / 50-70-90% CT sequence
TRAIN_CT_INPUT = load_config()["train_ct_input"]
TEST_CT_INPUT = load_config()["test_ct_input"]
MODELS = Path(__file__).resolve().parent
SOURCES = {
    "image_2d": MODELS / "Time_series_seq2seq_4DCT_voxelmorph.py",
    "rpm_1d": MODELS / "RPM_Time_series_seq2seq_4DCT_voxelmorph.py",
    "no_surrogate": MODELS / "seq2seq_4DCT_voxelmorph.py",
}


def settings(method):
    """Flattened training configuration for one method (from ``configs/config.yaml``)."""
    if method not in METHODS or {TRAIN_CT_INPUT, TEST_CT_INPUT} - set(CT_INPUTS):
        raise ValueError((method, TRAIN_CT_INPUT, TEST_CT_INPUT))
    config = load_config()
    return dict(**config["methods"][method], **config["optimizer"],
                scheduler=dict(config["scheduler"]), epochs=config["epochs"],
                batch_size=config["batch_size"], channels=config["channels"], seed=config["seed"],
                lambda_smooth=config["lambda_smooth"], early_stopping=False,
                train_ct_input=TRAIN_CT_INPUT, test_ct_input=TEST_CT_INPUT,
                warped_volume="Phase 1 (50%) CT")


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
    # The original model files use bare imports (``from layers import ...``).
    sys.path.insert(0, str(MODELS))
    try:
        spec = importlib.util.spec_from_file_location(f"original_{method}", SOURCES[method])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODELS))
    model = module.EncoderDecoderConvLSTM(channels, 1, size, size, size)
    if recompute:
        # Checkpoints from the reported runs were saved with these wrappers.
        for name in ("ConvLSTM3d1", "ConvLSTM3d2", "ConvLSTM3d3", "ConvLSTM3d4"):
            setattr(model, name, Recompute(getattr(model, name)))
    return model


def make_optimizer(model, method):
    config = settings(method)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                 eps=config["eps"], weight_decay=config["weight_decay"])
    return optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["scheduler"])


def tensors(case, device, method, training=False, ct_input=TEST_CT_INPUT):
    """Return (ct, surrogate, rpm, target); ct is (1,3,1,D,H,W) for "sequence", (1,1,1,D,H,W) for "reference"."""
    if ct_input not in CT_INPUTS:
        raise ValueError(ct_input)
    target = torch.from_numpy(case["target"])[None, None].to(device)
    probability = settings(method)["per_axial_slice_flip_probability"]
    if training and probability > 0:
        # The original no-surrogate pipeline applied RandomHorizontalFlip independently to
        # every raw axial slice, including the ten discarded leading slices; reproduced here.
        t, d = target.shape[2:4]
        flips = (torch.rand(t, d + 10)[:, 10:] < probability).to(device)[None, None, :, :, None, None]
        target = torch.where(flips, target.flip(-1), target)
    ct = target.transpose(1, 2).contiguous().clone().detach()
    if ct_input == "reference":
        ct = ct[:, :1].contiguous()
    surrogate = torch.from_numpy(case["surrogate"])[None].to(device)
    rpm = torch.from_numpy(case["rpm"])[None].to(device)
    return ct, surrogate, rpm, target


def patient_order(patients, method):
    if settings(method)["shuffle"]:
        return [patients[i] for i in torch.randperm(len(patients)).tolist()]
    return list(patients)


def predict(model, method, ct, surrogate=None, rpm=None):
    """Return (prediction (B,1,3,D,H,W), DVF (B,3,3,D,H,W)); ct is (B,3,1,D,H,W) or (B,1,1,D,H,W).

    ``ct[:, 0]`` must be the 50% reference; it is the volume warped by every predicted DVF.
    """
    if ct.ndim != 6 or ct.shape[1] not in (1, 3) or ct.shape[2] != 1:
        raise ValueError("expected a CT sequence (B,3,1,D,H,W) or reference (B,1,1,D,H,W)")
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
