# sim-lib-numbers-signal

In one line: It transforms, combines, estimates, aligns, and safely unmixes sampled patterns with every convention stated.

## What it gives you

This library separates a signal into Fourier, cosine, or sine components and
puts it back together again. It also convolves, cross-correlates, and performs
regularized deconvolution over canonical real tensors. Direct definitions stay
beside FFT paths, and automatic convolution exposes the cost comparison that
made its choice. Linear full, same, and valid spans cannot be confused with
circular geometry; boundary, normalization, and lag-order policies remain
typed and reviewable. Overlap-add and overlap-save plans state their retained
span, latency, padding, and discarded boundaries. Deconvolution always carries
a Tikhonov or truncated spectral guard plus singular-bin and residual evidence.

For classical spectral estimation it supplies rectangular, Hann, Hamming,
Blackman-family, Kaiser, and caller-defined windows with explicit endpoint and
normalization policy. Periodogram, Welch, cross-spectrum/coherence, Slepian
multitaper, and uneven-sample Lomb-Scargle reports retain their frequency grid,
scaling denominator, degrees of freedom, segment/taper counts, and admitted
work ceiling. A result can therefore be reconstructed without guessing whether
power was folded, gain-corrected, density-scaled, or variance-normalized.

Burg autoregressive fits reject singular or unstable reflection stages by
default, can deliberately reduce to the last stable order, and select fixed,
AIC, BIC, or final-prediction-error order with every candidate score retained.
Maximum-entropy spectra reuse the same bounded frequency-grid evidence, while
forward and backward predictions enforce horizon, work, and amplitude ceilings.
Complete DFT bins can be evaluated or integrated between sample points under an
explicit origin, period, endpoint, wrapping, Nyquist, sign, and normalization
contract. Single-bin evaluation, Hilbert analytic signals, phase unwrapping,
instantaneous frequency, and attack/release envelopes use those conventions too.

Multilevel Haar and Le Gall 5/3 wavelets preserve the chosen periodic,
symmetric, or zero boundary policy and retain every odd reconstruction length.
Savitzky-Golay filters expose their polynomial fit and always scale derivatives
by factorial and physical sample spacing. Structured Toeplitz inputs use shared
scaled-pivot linear algebra and return pivot conditioning and residual evidence.
Linear, natural-cubic, and monotone-cubic interpolation make duplicate samples
and reject, clamp, or linear extrapolation explicit.

The transform side handles ordinary real samples and paired complex samples,
including awkward prime lengths, with explicit scaling, direction, packing,
padding, stride, and output placement. Multidimensional views preserve declared
axes and physical strides, while bounded plans transform tensors larger than
memory through a caller-selected Table or Dir block store.

## Why you will be glad

- You can compare optimized frequency analysis with a clear reference answer.
- Prime-sized and composite signals follow the same deterministic contract.
- Explicit conventions prevent silent sign, scaling, and packing mismatches.
- Direct and FFT convolution agree under one typed mode and crop contract.
- Correlation returns signed lags instead of leaving index meaning implicit.
- Singular deconvolution returns finite conditioning evidence, never infinity.
- Blocked convolution reports exactly what it retained, padded, and discarded.
- Tone and noise estimates retain window, grid, scaling, and averaging evidence.
- Burg/MEM results retain stable order, residual, reflection, and criterion evidence.
- Periodic interpolation and integration state every grid and endpoint assumption.
- Hilbert phase, instantaneous frequency, and envelopes share one Fourier contract.
- Odd multilevel wavelets reconstruct under every declared boundary mode.
- Savitzky-Golay derivatives preserve fitted polynomials in physical units.
- Toeplitz solves report singular pivots or finite residual evidence.
- Sample interpolation types duplicate and extrapolation behavior.
- Segment, taper, frequency-grid, and total-work ceilings fail before execution.
- Canonical tensor buffers let the result move directly into other number work.
- Scratch, passes, block I/O, precision, and a content digest make external
  plans reviewable before and after execution.

## Where it fits

This is the reusable signal-algorithm layer in the SIM number stack. Audio,
image, numerical, and scientific libraries can share it without embedding
their own Fourier convention, convolution threshold, lag convention, unsafe
spectral division, private complex storage, or filesystem-specific out-of-core
API. The loadable Lisp surface exposes `signal/transform`, `signal/convolve`,
`signal/correlate`, `signal/deconvolve`, `signal/burg`, and
`signal/dft-interpolate` with the same explicit policy.
