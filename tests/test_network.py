import pytest
import torch

from demt_li.network import build_model, predict


@pytest.mark.parametrize("method", ["image_2d", "rpm_1d", "no_surrogate"])
def test_forward_shapes(method):
    torch.manual_seed(0)
    size = 16
    model = build_model(method, size=size, channels=4).eval()
    ct = torch.randn(1, 3, 1, size, size, size)
    surrogate = torch.randn(1, 3, 3, size, size)
    rpm = torch.tensor([[28., 28., 25.]])
    with torch.no_grad():
        prediction, flow = predict(model, method, ct, surrogate, rpm)
    assert prediction.shape == (1, 1, 3, size, size, size)
    assert flow.shape == (1, 3, 3, size, size, size)


def test_checkpoint_parameter_count():
    assert sum(p.numel() for p in build_model("image_2d").parameters()) == 9112675
