"""Simplified solar-coronal and residual-light background models."""

from __future__ import annotations

import torch
from .spectra import solar_spectrum_proxy


def solar_corona_spectrum(
    wavelengths_nm: torch.Tensor,
    reference_level: float = 1.0,
    spectral_index: float = -2.0,
    reference_wavelength_nm: float = 500.0,
) -> torch.Tensor:
    wl = wavelengths_nm.to(torch.float64)
    return reference_level * solar_spectrum_proxy(wl) * (
        wl / reference_wavelength_nm
    ) ** spectral_index


def residual_solar_leakage(
    wavelengths_nm: torch.Tensor,
    leakage_fraction: float = 1.0e-8,
) -> torch.Tensor:
    if leakage_fraction < 0:
        raise ValueError("leakage_fraction must be nonnegative")
    return leakage_fraction * solar_spectrum_proxy(wavelengths_nm)


def total_background_spectrum(
    wavelengths_nm: torch.Tensor,
    *,
    corona_level: float = 1.0,
    coronal_spectral_index: float = -2.0,
    leakage_fraction: float = 1.0e-8,
) -> torch.Tensor:
    return solar_corona_spectrum(
        wavelengths_nm,
        reference_level=corona_level,
        spectral_index=coronal_spectral_index,
    ) + residual_solar_leakage(wavelengths_nm, leakage_fraction)
