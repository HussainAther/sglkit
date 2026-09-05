import torch
from sglkit.diffraction import make_monopole_psf, monopole_sgl_psf, radial_grid


def test_monopole_psf_finite_nonnegative_and_normalized():
    psf = make_monopole_psf(size=65, pixel_scale_m=0.05)
    assert torch.isfinite(psf).all()
    assert torch.all(psf >= 0)
    assert torch.isclose(psf.max(), torch.tensor(1.0, dtype=psf.dtype))


def test_center_is_peak():
    psf = make_monopole_psf(size=65, pixel_scale_m=0.05)
    c = psf.shape[0] // 2
    assert torch.isclose(psf[c, c], psf.max())


def test_wavelength_changes_response():
    rho = radial_grid(33, 0.05)
    a = monopole_sgl_psf(rho, 500e-9, 650 * 149_597_870_700.0)
    b = monopole_sgl_psf(rho, 1000e-9, 650 * 149_597_870_700.0)
    assert not torch.allclose(a, b)
