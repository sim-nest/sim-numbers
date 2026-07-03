# sim-lib-numbers-tensor-bit

In one line: It stores big grids of yes-or-no values tightly packed and combines them with logical operations.

## What it gives you

When you have a large grid where every cell is simply on or off, storing each as a full number wastes room. This packs those true-or-false cells together so they take very little space, then lets you combine whole grids with the everyday logical operations: keep where both are on, keep where either is on, keep where they differ. It is built for masks, flags, and membership grids where the answer per cell is binary. If two grids do not have matching shapes, it refuses rather than quietly producing a misaligned result, so mistakes surface instead of hiding.

## Why you will be glad

- Huge on-or-off grids take a fraction of the usual storage.
- Combining masks with and, or, and exclusive-or is direct and quick.
- Mismatched shapes are rejected outright, so silent misalignment cannot slip through.

## Where it fits

This is the packed boolean specialization of the SIM tensor stack. It plugs into the shared grid interface as the backend for true-or-false element data, giving the constellation an efficient home for masks and logical grids alongside the numeric specializations.
