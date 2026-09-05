"""Simplified deterministic SGL imaging and raster mission simulation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .acquisition import DetectorConfig, add_pointing_and_navigation_error, apply_detector_response
from .diffraction import make_monopole_psf


def simulate_sgl_image(
    planet_map: torch.Tensor,
    *,
    wavelength_m: float = 1.0e-6,
    z_au: float = 650.0,
    pixel_scale_m: float = 0.05,
) -> torch.Tensor:
    """Convolutional scalar-monopole approximation for an extended source."""
    if planet_map.ndim != 2:
        raise ValueError("planet_map must be 2-D")
    kernel = make_monopole_psf(
        size=planet_map.shape[-1],
        pixel_scale_m=pixel_scale_m,
        wavelength_m=wavelength_m,
        z_au=z_au,
        normalize="sum",
        device=planet_map.device,
        dtype=planet_map.dtype,
    )
    image = F.conv2d(
        planet_map[None, None],
        kernel[None, None],
        padding="same",
    )
    return image[0, 0]


def sample_image_at_positions(
    image: torch.Tensor,
    positions_xy_m: torch.Tensor,
    pixel_scale_m: float,
) -> torch.Tensor:
    center = (image.shape[-1] - 1) / 2.0
    x = torch.round(positions_xy_m[:, 0] / pixel_scale_m + center).long()
    y = torch.round(positions_xy_m[:, 1] / pixel_scale_m + center).long()
    x = torch.clamp(x, 0, image.shape[-1] - 1)
    y = torch.clamp(y, 0, image.shape[-2] - 1)
    return image[y, x]


def simulate_raster_observations(
    *,
    planet_map: torch.Tensor,
    raster_positions_xy_m: torch.Tensor,
    detector: DetectorConfig,
    integration_time_s: float,
    planet_rate_scale: float = 1.0,
    background_rate: float = 0.0,
    pixel_scale_m: float = 0.05,
    wavelength_m: float = 1.0e-6,
    z_au: float = 650.0,
    planet_throughput: float = 1.0,
    background_throughput: float = 1.0,
    pointing_sigma_m: float = 0.0,
    navigation_sigma_m: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    sgl_image = simulate_sgl_image(
        planet_map,
        wavelength_m=wavelength_m,
        z_au=z_au,
        pixel_scale_m=pixel_scale_m,
    )
    actual_positions = add_pointing_and_navigation_error(
        raster_positions_xy_m,
        pointing_sigma_m,
        navigation_sigma_m,
        generator,
    )
    sampled = sample_image_at_positions(sgl_image, actual_positions, pixel_scale_m)
    planet_rate = sampled * planet_rate_scale * planet_throughput
    total_rate = planet_rate + background_rate * background_throughput
    measurements = apply_detector_response(total_rate, detector, integration_time_s, generator)
    return {
        "ideal_positions_xy_m": raster_positions_xy_m,
        "actual_positions_xy_m": actual_positions,
        "sgl_image": sgl_image,
        "planet_rate": planet_rate,
        "total_rate": total_rate,
        "measurements": measurements,
    }
