import torch
from sglkit.acquisition import DetectorConfig, apply_detector_response, make_square_raster


def test_square_raster_shape():
    raster = make_square_raster(1.0, 1.0)
    assert raster.ndim == 2
    assert raster.shape[1] == 2


def test_detector_response_finite():
    rate = torch.ones(10, dtype=torch.float64)
    detector = DetectorConfig()
    g = torch.Generator().manual_seed(1)
    out = apply_detector_response(rate, detector, 10.0, g)
    assert torch.isfinite(out).all()
