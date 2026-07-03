# sim-lib-numbers-tensor-cmplxf

In one line: It holds grids of complex numbers efficiently, for signal and wave work done in bulk.

## What it gives you

Fields like signal processing and physics work with whole arrays of complex numbers, each having a real and an imaginary part. This provides a grid specialized to carry exactly those two-part values compactly and to operate on them as a unit. Instead of scattering the pairs loosely, it keeps them in a tight, ordered arrangement suited to fast bulk work, and it converts cleanly to and from the system's general grid form so it stays fully compatible. If the number of values does not match the declared shape, it refuses rather than guess, so your data stays trustworthy.

## Why you will be glad

- Large collections of complex values are stored compactly and handled together.
- Signal and wave computations over whole arrays become natural to express.
- Shape and data are kept consistent, with mismatches caught instead of masked.

## Where it fits

This is the complex-number specialization within the SIM tensor stack. It registers against the shared grid interface as the backend for two-part element data, letting the constellation carry complex-valued arrays with the same uniformity as its other grid types.
