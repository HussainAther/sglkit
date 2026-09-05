import torch
from sglkit.acquisition import DetectorConfig, make_square_raster
from sglkit.mission import simulate_sgl_image, simulate_raster_observations


def test_simulate_sgl_image_shape_and_finite():
    planet = torch.zeros((33, 33), dtype=torch.float64)
    planet[16, 16] = 1.0
    image = simulate_sgl_image(planet)
    assert image.shape == planet.shape
    assert torch.isfinite(image).all()


def test_raster_observations_keys():
    planet = torch.zeros((33, 33), dtype=torch.float64)
    planet[16, 16] = 1.0
    raster = make_square_raster(0.5, 0.5)
    out = simulate_raster_observations(
        planet_map=planet,
        raster_positions_xy_m=raster,
        detector=DetectorConfig(),
        integration_time_s=10.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert out["measurements"].shape[0] == raster.shape[0]
