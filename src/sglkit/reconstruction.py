"""Baseline and forward-model-aware SGL reconstruction methods."""

from __future__ import annotations

import torch

from .mission import sample_image_at_positions, simulate_sgl_image


def reconstruct_nearest_raster(
    raster_positions_xy_m: torch.Tensor,
    measurements: torch.Tensor,
    *,
    output_size: int,
    pixel_scale_m: float,
) -> torch.Tensor:
    image = torch.zeros((output_size, output_size), dtype=measurements.dtype, device=measurements.device)
    counts = torch.zeros_like(image)
    center = (output_size - 1) / 2.0
    x = torch.round(raster_positions_xy_m[:, 0] / pixel_scale_m + center).long()
    y = torch.round(raster_positions_xy_m[:, 1] / pixel_scale_m + center).long()
    valid = (x >= 0) & (x < output_size) & (y >= 0) & (y < output_size)
    for xi, yi, value in zip(x[valid], y[valid], measurements[valid]):
        image[yi, xi] += value
        counts[yi, xi] += 1
    mask = counts > 0
    image[mask] = image[mask] / counts[mask]
    return image


def total_variation(image: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(image[1:, :] - image[:-1, :]).mean()
    dx = torch.abs(image[:, 1:] - image[:, :-1]).mean()
    return dx + dy


def forward_raster_model(
    source: torch.Tensor,
    raster_positions_xy_m: torch.Tensor,
    *,
    planet_rate_scale: float = 1.0,
    pixel_scale_m: float = 0.05,
    wavelength_m: float = 1.0e-6,
    z_au: float = 650.0,
) -> torch.Tensor:
    image = simulate_sgl_image(
        source,
        wavelength_m=wavelength_m,
        z_au=z_au,
        pixel_scale_m=pixel_scale_m,
    )
    return sample_image_at_positions(image, raster_positions_xy_m, pixel_scale_m) * planet_rate_scale


def reconstruct_tv(
    calibrated_measurements: torch.Tensor,
    raster_positions_xy_m: torch.Tensor,
    *,
    output_size: int,
    planet_rate_scale: float = 1.0,
    pixel_scale_m: float = 0.05,
    wavelength_m: float = 1.0e-6,
    z_au: float = 650.0,
    iterations: int = 200,
    learning_rate: float = 0.05,
    tv_weight: float = 1.0e-4,
) -> tuple[torch.Tensor, list[float]]:
    """Recover a nonnegative image using the same simplified SGL forward operator.

    The latent image is sigmoid-parameterized to keep values in [0, 1].
    """
    latent = torch.zeros(
        (output_size, output_size),
        dtype=calibrated_measurements.dtype,
        device=calibrated_measurements.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    history: list[float] = []
    scale = torch.mean(calibrated_measurements**2).detach().clamp_min(1.0e-12)

    for _ in range(iterations):
        optimizer.zero_grad()
        source = torch.sigmoid(latent)
        prediction = forward_raster_model(
            source,
            raster_positions_xy_m,
            planet_rate_scale=planet_rate_scale,
            pixel_scale_m=pixel_scale_m,
            wavelength_m=wavelength_m,
            z_au=z_au,
        )
        data_loss = torch.mean((prediction - calibrated_measurements) ** 2) / scale
        loss = data_loss + tv_weight * total_variation(source)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    return torch.sigmoid(latent).detach(), history
