import torch
from sglkit.filters import biomimetic_filter, integrated_throughput
from sglkit.spectra import reflected_planet_spectrum, coronal_scattered_light_proxy

wl = torch.linspace(300.0, 1000.0, 1000, dtype=torch.float64)
planet = reflected_planet_spectrum(wl)
background = coronal_scattered_light_proxy(wl, spectral_index=-3.0)
t = biomimetic_filter(wl, cutoff_nm=442.0, absorption_depth=0.20)

print("Planet throughput:", integrated_throughput(planet, t, wl))
print("Background throughput:", integrated_throughput(background, t, wl))
