from pathlib import Path
import matplotlib.pyplot as plt
from sglkit.diffraction import make_monopole_psf

psf = make_monopole_psf(size=501, pixel_scale_m=0.02, wavelength_m=1e-6, z_au=650)
row = psf[psf.shape[0] // 2].cpu().numpy()
coords = ((range(psf.shape[0])))
coords = [(i - (psf.shape[0] - 1) / 2) * 0.02 for i in coords]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(coords, row)
ax.set_xlim(-2, 2)
ax.set_xlabel("Image-plane displacement (m)")
ax.set_ylabel("Normalized intensity")
ax.set_title("Idealized scalar monopole SGL PSF\n1 micrometer, 650 AU")
ax.grid(alpha=0.25)
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/monopole_psf.png", dpi=200, bbox_inches="tight")
