import numpy as np
import torch
from PIL import Image

from demt_li.preprocessing import load_png_sequence, preprocess_ct_slice, read_rpm
from demt_li.regional_input import CONDITIONS, regional_input


def test_ct_slice_crop_and_resize():
    image = torch.zeros(1, 512, 512)
    image[:, 66:386, 96:416] = 100.  # the 320 x 320 crop window
    out = preprocess_ct_slice(image)
    assert out.shape == (1, 128, 128)
    assert torch.allclose(out, torch.full_like(out, 100.))


def test_png_sequence_shape(tmp_path):
    for phase in (50, 70, 90):
        Image.fromarray(np.full((64, 64, 4), 128, np.uint8)).save(tmp_path / f"{phase}.png")
    assert load_png_sequence(tmp_path).shape == (3, 3, 128, 128)


def test_rpm_columns(tmp_path):
    path = tmp_path / "rpm.csv"
    path.write_text("﻿" + "\n".join(",".join(str(10 * r + c) for c in range(10)) for r in range(6)))
    assert read_rpm(path, 2).tolist() == [15., 17., 19.]


def test_regional_conditions():
    x = np.random.default_rng(0).normal(size=(3, 3, 128, 128)).astype(np.float32)
    assert len(CONDITIONS) == 25
    assert np.array_equal(regional_input(x, "full"), x)
    for condition in CONDITIONS[1:]:
        y = regional_input(x, condition)
        assert y.shape == x.shape and np.isfinite(y).all()
