import torch

from project.models.losses import dvf_smoothness, objective


def test_constant_field_has_zero_smoothness():
    assert dvf_smoothness(torch.full((1, 3, 3, 4, 4, 4), 2.5)).item() == 0


def test_linear_ramp_gradient():
    # u_x = x along the last axis: gradient 1 in one of 9 (component x axis) combinations.
    x = torch.arange(5.).expand(1, 1, 1, 5, 5, 5)
    flow = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
    assert torch.isclose(dvf_smoothness(flow), torch.tensor(1 / 3))


def test_objective_is_mse_plus_unit_weighted_smoothness():
    prediction = torch.zeros(1, 1, 3, 4, 4, 4)
    target = torch.ones(1, 1, 3, 4, 4, 4)
    flow = torch.randn(1, 3, 3, 4, 4, 4)
    loss, mse, smooth = objective(prediction, target, flow)
    assert mse.item() == 1
    assert torch.isclose(loss, mse + smooth)
