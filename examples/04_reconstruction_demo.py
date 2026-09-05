import torch
from sglkit.acquisition import make_square_raster
from sglkit.reconstruction import forward_raster_model, reconstruct_tv
from sglkit.metrics import mse, normalize_image, psnr

truth = torch.zeros((33, 33), dtype=torch.float64)
truth[9:18, 10:22] = 1.0
truth[20:26, 8:16] = 0.6
raster = make_square_raster(0.75, 0.15)
y = forward_raster_model(truth, raster, planet_rate_scale=5.0, pixel_scale_m=0.05)
recon, history = reconstruct_tv(
    y,
    raster,
    output_size=33,
    planet_rate_scale=5.0,
    pixel_scale_m=0.05,
    iterations=100,
)
truth_n = normalize_image(truth)
recon_n = normalize_image(recon)
print("Final loss:", history[-1])
print("MSE:", mse(truth_n, recon_n))
print("PSNR:", psnr(truth_n, recon_n))
