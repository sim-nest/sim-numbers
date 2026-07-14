# sim-lib-numbers-tensor-f64

In one line: It gives you fast grids of ordinary decimal numbers, the common case for number-heavy work.

## What it gives you

Most grid math, from data analysis to simulation, runs on plain decimal numbers. This provides a grid specialized to hold those decimals in a single tight, contiguous block and to run math across them at speed. Because the values sit together in memory rather than scattered, operations sweep through them quickly. It converts cleanly to and from the system's general grid form, so this fast version and the uniform one use the same interface. If the number of values you supply does not fit the shape you declared, it declines rather than proceed with mismatched data, keeping results sound.

## Why you will be glad

- Decimal grid math runs quickly thanks to tightly-packed storage.
- The common case for numeric arrays is handled with no extra ceremony.
- Shape errors are caught up front instead of corrupting the result quietly.

## Where it fits

This is the everyday decimal specialization of the SIM tensor stack, and likely the most-used backend. It plugs into the shared grid interface, giving the constellation a fast footing for the large decimal arrays that ordinary numeric work depends on.
