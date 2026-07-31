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
`signal/correlate`, and `signal/deconvolve` with the same explicit policy.
