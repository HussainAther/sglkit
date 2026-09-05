"""Common image and photometric metrics."""

from __future__ import annotations

import math
import torch


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    lo = image.min()
    hi = image.max()
    if hi <= lo:
        return torch.zeros_like(image)
    return (image - lo) / (hi - lo)


def mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(torch.mean((reference - estimate) ** 2).item())


def psnr(reference: torch.Tensor, estimate: torch.Tensor, data_range: float = 1.0) -> float:
    err = mse(reference, estimate)
    if err == 0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / err)


def photometric_snr(signal_e: float, background_e: float, read_noise_e: float = 0.0) -> float:
    variance = signal_e + background_e + read_noise_e**2
    return 0.0 if variance <= 0 else signal_e / math.sqrt(variance)
