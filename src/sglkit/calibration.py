"""Background subtraction and throughput calibration helpers."""

from __future__ import annotations

import torch


def expected_background_electrons(
    *,
    background_rate: float,
    background_throughput: float,
    integration_time_s: float,
    quantum_efficiency: float,
    dark_current_e_per_s: float = 0.0,
) -> float:
    optical = background_rate * background_throughput * quantum_efficiency * integration_time_s
    dark = dark_current_e_per_s * integration_time_s
    return optical + dark


def calibrate_measurements(
    measurements_adu: torch.Tensor,
    *,
    background_rate: float,
    background_throughput: float,
    integration_time_s: float,
    quantum_efficiency: float,
    gain_e_per_adu: float = 1.0,
    dark_current_e_per_s: float = 0.0,
    planet_throughput: float = 1.0,
    clip_negative: bool = True,
) -> torch.Tensor:
    if integration_time_s <= 0 or quantum_efficiency <= 0 or gain_e_per_adu <= 0 or planet_throughput <= 0:
        raise ValueError("integration time, QE, gain, and throughput must be positive")
    electrons = measurements_adu * gain_e_per_adu
    background_e = expected_background_electrons(
        background_rate=background_rate,
        background_throughput=background_throughput,
        integration_time_s=integration_time_s,
        quantum_efficiency=quantum_efficiency,
        dark_current_e_per_s=dark_current_e_per_s,
    )
    signal_e = electrons - background_e
    if clip_negative:
        signal_e = torch.clamp(signal_e, min=0.0)
    return signal_e / (quantum_efficiency * integration_time_s * planet_throughput)
