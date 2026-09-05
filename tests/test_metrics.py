import math
import torch
from sglkit.metrics import mse, normalize_image, psnr


def test_mse_zero_and_psnr_inf_for_identity():
    x = torch.tensor([[0.0, 1.0]])
    assert mse(x, x) == 0.0
    assert math.isinf(psnr(x, x))


def test_normalize_bounds():
    x = torch.tensor([[2.0, 4.0]])
    y = normalize_image(x)
    assert y.min() == 0
    assert y.max() == 1
