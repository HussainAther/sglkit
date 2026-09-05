# sglkit

`sglkit` is a small Python toolkit for reproducible Solar Gravitational Lens (SGL) imaging experiments.

It provides building blocks for:

- scalar monopole SGL diffraction models,
- simple exoplanet and solar-coronal spectral proxies,
- wavelength-dependent optical filtering,
- detector and raster-acquisition simulation,
- background calibration and throughput correction,
- baseline and iterative image reconstruction,
- Monte Carlo metrics and benchmark result export.

## Scientific scope

This package is intended for computational feasibility studies, sensitivity analysis, method development, and reproducible research. The current forward model is a **simplified scalar monopole approximation**. It is not a full vector SGL propagation model and does not yet include solar multipole moments, a calibrated coronagraph model, or mission-qualified detector/systematics models.

## Installation

```bash
pip install -e .
```

For plotting and tests:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from sglkit.diffraction import make_monopole_psf
from sglkit.filters import biomimetic_filter

psf = make_monopole_psf(
    size=257,
    pixel_scale_m=0.05,
    wavelength_m=1e-6,
    z_au=650.0,
)

wavelengths_nm = torch.linspace(300.0, 1000.0, 1000)
transmission = biomimetic_filter(
    wavelengths_nm,
    cutoff_nm=442.0,
    absorption_depth=0.20,
)
```

See `examples/` for end-to-end demonstrations.

## Package layout

- `diffraction.py`: scalar monopole SGL point-spread function
- `spectra.py`: solar/exoplanet spectral proxies
- `filters.py`: generic and biomimetic spectral filters
- `corona.py`: simplified solar-coronal background models
- `acquisition.py`: detector and raster sampling models
- `mission.py`: deterministic SGL image and raster simulation
- `calibration.py`: background subtraction and throughput correction
- `reconstruction.py`: baseline and TV-regularized inverse reconstruction
- `metrics.py`: MSE, PSNR, SNR, and summaries
- `benchmarks.py`: machine-readable benchmark records

## Reproducibility

The project includes deterministic seeds, explicit configuration parameters, tests, and GitHub Actions. Paper-specific experiments should live in a separate research repository and depend on a tagged release of `sglkit`.

## Citation

If you use this software in academic work, please cite the repository release. See `CITATION.cff`.

## License

MIT.
