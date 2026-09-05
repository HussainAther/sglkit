import torch
from sglkit.calibration import calibrate_measurements


def test_noiseless_calibration_recovers_signal_rate():
    signal_rate = torch.tensor([2.0], dtype=torch.float64)
    background_rate = 3.0
    qe = 0.8
    t = 10.0
    planet_throughput = 0.5
    background_throughput = 0.25
    electrons = (
        signal_rate * planet_throughput * qe * t
        + background_rate * background_throughput * qe * t
    )
    recovered = calibrate_measurements(
        electrons,
        background_rate=background_rate,
        background_throughput=background_throughput,
        integration_time_s=t,
        quantum_efficiency=qe,
        planet_throughput=planet_throughput,
    )
    assert torch.allclose(recovered, signal_rate)
