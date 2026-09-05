"""Optional plotting helpers. Requires matplotlib."""

from __future__ import annotations


def plot_psf(psf, *, extent_m=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    extent = None
    if extent_m is not None:
        extent = [-extent_m, extent_m, -extent_m, extent_m]
    image = ax.imshow(psf.detach().cpu().numpy(), origin="lower", extent=extent)
    ax.set_title("Scalar monopole SGL PSF")
    ax.set_xlabel("x (m)" if extent_m is not None else "pixel")
    ax.set_ylabel("y (m)" if extent_m is not None else "pixel")
    return ax, image
