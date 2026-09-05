"""Spectral filter models."""

from __future__ import annotations

import torch


def biomimetic_filter(
    wavelengths_nm: torch.Tensor,
    cutoff_nm: float = 442.0,
    absorption_depth: float = 0.20,
    transition_width_nm: float = 6.0,
) -> torch.Tensor:
    """Phenomenological carotenoprotein-inspired short-wavelength filter."""
    wl = wavelengths_nm.to(torch.float64)
    base = 1.0 / (1.0 + torch.exp(-(wl - cutoff_nm) / transition_width_nm))
    attenuation = torch.ones_like(wl)
    for peak_nm in (384.0, 405.0, 433.0, 458.0):
        attenuation -= absorption_depth * torch.exp(
            -0.5 * ((wl - peak_nm) / 5.0) ** 2
        )
    return torch.clamp(base * attenuation, 0.02, 1.0)


def apply_filter(spectrum: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
    if spectrum.shape != transmission.shape:
        raise ValueError("spectrum and transmission must have the same shape")
    return spectrum * transmission


def integrated_throughput(
    spectrum: torch.Tensor,
    transmission: torch.Tensor,
    wavelengths_nm: torch.Tensor,
) -> float:
    denominator = torch.trapz(spectrum, wavelengths_nm)
    if denominator <= 0:
        raise ValueError("spectrum must have positive integrated flux")
    numerator = torch.trapz(spectrum * transmission, wavelengths_nm)
    return float((numerator / denominator).item())
