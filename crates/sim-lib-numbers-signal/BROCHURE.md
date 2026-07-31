# sim-lib-numbers-signal

In one line: It turns sampled patterns into trustworthy frequency and symmetry views with every convention stated.

## What it gives you

This library separates a signal into Fourier, cosine, or sine components and
puts it back together again. It handles ordinary real samples and paired
complex samples, including awkward prime lengths, while making scaling,
direction, packing, padding, stride, and output placement explicit. Direct
definitions stay available beside the faster paths, so a result can be checked
against the mathematics that defines it.

## Why you will be glad

- You can compare optimized frequency analysis with a clear reference answer.
- Prime-sized and composite signals follow the same deterministic contract.
- Explicit conventions prevent silent sign, scaling, and packing mismatches.
- Canonical tensor buffers let the result move directly into other number work.

## Where it fits

This is the reusable one-dimensional signal-transform layer in the SIM number
stack. Audio, image, numerical, and scientific libraries can share it without
embedding their own Fourier convention or private complex storage.
