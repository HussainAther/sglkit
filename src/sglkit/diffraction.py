"""Simplified scalar monopole Solar Gravitational Lens diffraction model."""

from __future__ import annotations

import math
import torch

G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
AU_M = 149_597_870_700.0
R_G = 2.0 * G * M_SUN / C**2


def monopole_sgl_psf(
    rho_m: torch.Tensor,
    wavelength_m: float,
    z_m: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the idealized scalar monopole SGL intensity response J0^2."""
    if wavelength_m <= 0 or z_m <= 0:
        raise ValueError("wavelength_m and z_m must be positive")
    k = 2.0 * math.pi / wavelength_m
    argument = k * rho_m * math.sqrt(2.0 * R_G / z_m)
    psf = torch.special.bessel_j0(argument) ** 2
    if normalize:
        peak = psf.max()
        if peak > 0:
            psf = psf / peak
    return psf


def radial_grid(
    size: int,
    pixel_scale_m: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a square physical-radius grid centered on the image."""
    if size <= 0 or pixel_scale_m <= 0:
        raise ValueError("size and pixel_scale_m must be positive")
    coords = (
        torch.arange(size, device=device, dtype=dtype)
        - (size - 1) / 2.0
    ) * pixel_scale_m
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.sqrt(x**2 + y**2)


def make_monopole_psf(
    size: int = 257,
    pixel_scale_m: float = 0.05,
    wavelength_m: float = 1.0e-6,
    z_au: float = 650.0,
    *,
    normalize: str = "peak",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a 2-D scalar monopole SGL PSF.

    normalize may be ``peak`` (max=1), ``sum`` (sum=1), or ``none``.
    """
    rho = radial_grid(size, pixel_scale_m, device=device, dtype=dtype)
    psf = monopole_sgl_psf(rho, wavelength_m, z_au * AU_M, normalize=False)
    if normalize == "peak":
        return psf / psf.max()
    if normalize == "sum":
        total = psf.sum()
        return psf / total if total > 0 else psf
    if normalize == "none":
        return psf
    raise ValueError("normalize must be 'peak', 'sum', or 'none'")
