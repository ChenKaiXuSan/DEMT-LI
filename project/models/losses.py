"""Training objective: image MSE + lambda * spatial DVF-gradient smoothness (lambda = 1)."""
from __future__ import annotations

import torch

LOSS_DEFINITION = {
    "objective": "mean_phase_MSE + 1.0 * mean_phase_spatial_DVF_gradient_squared",
    "gradient": "torch.gradient over the three spatial axes (-3,-2,-1), unit grid spacing; "
                "centered interior and first-order one-sided boundary differences",
    "reduction": "spatial derivatives summed; batch, DVF components, voxels and phases averaged; "
                 "no temporal derivative",
    "lambda_smooth": 1.0,
}


def dvf_smoothness(flow):
    """Mean squared spatial gradient of a (B,3,T,D,H,W) displacement field."""
    if flow.ndim != 6 or flow.shape[1] != 3 or min(flow.shape[-3:]) < 2:
        raise ValueError("DVF must be (B,3,T,D,H,W) with nontrivial spatial dimensions")
    return sum(torch.gradient(flow, dim=axis, spacing=(1,), edge_order=1)[0].square().mean()
               for axis in (-3, -2, -1))


def objective(prediction, target, flow, lambda_smooth=1.):
    """Return (loss, mse, smoothness); prediction/target are (B,1,T,D,H,W)."""
    mse = torch.stack([torch.nn.functional.mse_loss(prediction[:, :, i], target[:, :, i])
                       for i in range(target.shape[2])]).mean()
    smooth = dvf_smoothness(flow)
    return mse + lambda_smooth * smooth, mse, smooth
