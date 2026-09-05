import torch
from sglkit.acquisition import DetectorConfig, make_square_raster
from sglkit.mission import simulate_raster_observations

planet = torch.zeros((65, 65), dtype=torch.float64)
planet[20:45, 22:43] = 1.0
raster = make_square_raster(1.5, 0.15)

result = simulate_raster_observations(
    planet_map=planet,
    raster_positions_xy_m=raster,
    detector=DetectorConfig(),
    integration_time_s=300.0,
    planet_rate_scale=5.0,
    background_rate=2.0,
    pointing_sigma_m=0.01,
    navigation_sigma_m=0.02,
    generator=torch.Generator().manual_seed(123),
)

print("Samples:", result["measurements"].numel())
print("Mean ADU:", float(result["measurements"].mean()))
