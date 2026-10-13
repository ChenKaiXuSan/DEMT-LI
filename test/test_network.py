import numpy as np
import pytest
import torch

from project.models.network import (DEFAULT_CT_INPUT, build_model, checkpoint_ct_input, predict, settings,
                             tensors)


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


@pytest.mark.parametrize("method", ["image_2d", "rpm_1d", "no_surrogate"])
def test_reference_only_forward(method):
    torch.manual_seed(0)
    size = 16
    model = build_model(method, size=size, channels=4).eval()
    reference = torch.randn(1, 1, 1, size, size, size)
    with torch.no_grad():
        prediction, flow = predict(model, method, reference, torch.randn(1, 3, 3, size, size),
                                   torch.tensor([[28., 28., 25.]]))
    assert prediction.shape == (1, 1, 3, size, size, size)
    assert flow.shape == (1, 3, 3, size, size, size)


def test_ct_input_modes():
    case = dict(target=np.random.default_rng(0).normal(size=(3, 8, 8, 8)).astype(np.float32),
                surrogate=np.zeros((3, 3, 8, 8), np.float32), rpm=np.zeros(3, np.float32))
    reference, _, _, target = tensors(case, "cpu", "image_2d")  # default is reference-only
    sequence = tensors(case, "cpu", "image_2d", ct_input="sequence")[0]
    assert DEFAULT_CT_INPUT == "reference" and settings("image_2d")["ct_input"] == "reference"
    assert reference.shape == (1, 1, 1, 8, 8, 8) and sequence.shape == (1, 3, 1, 8, 8, 8)
    assert torch.equal(reference[0, 0, 0], target[0, 0, 0]) and torch.equal(sequence[0, 2, 0], target[0, 0, 2])


def test_checkpoints_without_mode_are_sequence():
    assert checkpoint_ct_input(dict(config={})) == "sequence"
    assert checkpoint_ct_input(dict(config=dict(ct_input="reference"))) == "reference"
