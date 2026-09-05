import torch
from sglkit.acquisition import make_square_raster
from sglkit.reconstruction import forward_raster_model, reconstruct_nearest_raster, reconstruct_tv, total_variation


def test_total_variation_constant_zero():
    x = torch.ones((8, 8), dtype=torch.float64)
    assert float(total_variation(x)) == 0.0


def test_nearest_reconstruction_shape():
    raster = make_square_raster(0.5, 0.5)
    values = torch.ones(raster.shape[0], dtype=torch.float64)
    image = reconstruct_nearest_raster(raster, values, output_size=33, pixel_scale_m=0.05)
    assert image.shape == (33, 33)


def test_forward_model_shape():
    source = torch.zeros((17, 17), dtype=torch.float64)
    source[8, 8] = 1.0
    raster = make_square_raster(0.2, 0.2)
    y = forward_raster_model(source, raster, pixel_scale_m=0.05)
    assert y.shape[0] == raster.shape[0]


def test_tv_reconstruction_runs_and_is_bounded():
    source = torch.zeros((17, 17), dtype=torch.float64)
    source[6:11, 6:11] = 1.0
    raster = make_square_raster(0.2, 0.2)
    y = forward_raster_model(source, raster, pixel_scale_m=0.05)
    recon, history = reconstruct_tv(
        y,
        raster,
        output_size=17,
        pixel_scale_m=0.05,
        iterations=4,
        learning_rate=0.03,
    )
    assert len(history) == 4
    assert torch.isfinite(recon).all()
    assert torch.all((recon >= 0) & (recon <= 1))
