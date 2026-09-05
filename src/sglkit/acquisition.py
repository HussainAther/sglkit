"""Detector and raster-acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DetectorConfig:
    quantum_efficiency: float = 0.85
    read_noise_e: float = 3.0
    dark_current_e_per_s: float = 0.01
    gain_e_per_adu: float = 1.0


def make_square_raster(
    half_width_m: float,
    step_m: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if half_width_m <= 0 or step_m <= 0:
        raise ValueError("half_width_m and step_m must be positive")
    coords = torch.arange(
        -half_width_m,
        half_width_m + 0.5 * step_m,
        step_m,
        device=device,
        dtype=dtype,
    )
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)


def add_pointing_and_navigation_error(
    raster_xy_m: torch.Tensor,
    pointing_sigma_m: float = 0.0,
    navigation_sigma_m: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    total_sigma = (pointing_sigma_m**2 + navigation_sigma_m**2) ** 0.5
    if total_sigma == 0:
        return raster_xy_m.clone()
    noise = torch.randn(
        raster_xy_m.shape,
        generator=generator,
        dtype=raster_xy_m.dtype,
        device=raster_xy_m.device,
    )
    return raster_xy_m + total_sigma * noise


def apply_detector_response(
    photon_rate: torch.Tensor,
    detector: DetectorConfig,
    integration_time_s: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if integration_time_s <= 0:
        raise ValueError("integration_time_s must be positive")
    if detector.gain_e_per_adu <= 0:
        raise ValueError("gain_e_per_adu must be positive")
    signal_e = torch.clamp(photon_rate, min=0.0) * detector.quantum_efficiency * integration_time_s
    dark_e = detector.dark_current_e_per_s * integration_time_s
    expected_e = signal_e + dark_e
    poisson_e = torch.poisson(expected_e, generator=generator)
    read = detector.read_noise_e * torch.randn(
        poisson_e.shape,
        generator=generator,
        dtype=poisson_e.dtype,
        device=poisson_e.device,
    )
    return (poisson_e + read) / detector.gain_e_per_adu
