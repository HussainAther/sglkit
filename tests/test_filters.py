import torch
from sglkit.filters import biomimetic_filter, integrated_throughput


def test_filter_bounded():
    wl = torch.linspace(300.0, 1000.0, 1000)
    t = biomimetic_filter(wl)
    assert torch.all(t >= 0)
    assert torch.all(t <= 1)


def test_filter_transmits_red_more_than_blue():
    wl = torch.tensor([400.0, 800.0])
    t = biomimetic_filter(wl)
    assert t[1] > t[0]


def test_throughput_in_range():
    wl = torch.linspace(300.0, 1000.0, 1000)
    spectrum = torch.ones_like(wl)
    throughput = integrated_throughput(spectrum, biomimetic_filter(wl), wl)
    assert 0 < throughput <= 1
