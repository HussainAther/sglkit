"""Simple spectral proxy models for feasibility studies."""

from __future__ import annotations

import torch

H = 6.62607015e-34
C = 299_792_458.0
K_B = 1.380649e-23


def _normalize(x: torch.Tensor) -> torch.Tensor:
    m = x.max()
    return x / m if m > 0 else x


def planck_spectral_radiance(
    wavelengths_nm: torch.Tensor,
    temperature_k: float,
) -> torch.Tensor:
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    wl_m = wavelengths_nm.to(torch.float64) * 1.0e-9
    exponent = H * C / (wl_m * K_B * temperature_k)
    radiance = (2.0 * H * C**2 / wl_m**5) / torch.expm1(exponent)
    return _normalize(radiance)


def solar_spectrum_proxy(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    """Normalized 5772 K blackbody proxy for the solar spectrum."""
    return planck_spectral_radiance(wavelengths_nm, 5772.0)


def earth_like_albedo_proxy(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    """Phenomenological Earth-like albedo proxy, not an empirical spectrum."""
    wl = wavelengths_nm.to(torch.float64)
    albedo = torch.full_like(wl, 0.30)
    albedo += 0.12 * torch.exp(-0.5 * ((wl - 450.0) / 70.0) ** 2)
    albedo += 0.10 / (1.0 + torch.exp(-(wl - 700.0) / 18.0))
    albedo -= 0.07 * torch.exp(-0.5 * ((wl - 760.0) / 12.0) ** 2)
    albedo -= 0.08 * torch.exp(-0.5 * ((wl - 940.0) / 18.0) ** 2)
    return torch.clamp(albedo, 0.02, 0.95)


def reflected_planet_spectrum(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    return _normalize(
        solar_spectrum_proxy(wavelengths_nm)
        * earth_like_albedo_proxy(wavelengths_nm)
    )


def coronal_scattered_light_proxy(
    wavelengths_nm: torch.Tensor,
    spectral_index: float = -1.5,
) -> torch.Tensor:
    wl = wavelengths_nm.to(torch.float64)
    spectrum = solar_spectrum_proxy(wl) * (wl / 500.0) ** spectral_index
    return _normalize(spectrum)
