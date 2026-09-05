# Forward models

The initial SGL response is a scalar monopole intensity approximation proportional to `J0(argument)^2`.

For an extended source, `sglkit.mission.simulate_sgl_image` currently uses this kernel in a shift-invariant convolutional approximation. This is deliberately simple and should not be confused with a complete vector or multipole SGL propagation model.

Planned validation layers include analytic limiting cases, published wave-optical benchmarks, solar multipole corrections, coronagraph transfer functions, and instrument aberrations.
