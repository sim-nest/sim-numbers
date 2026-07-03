# sim-lib-numbers-tensor-rat64

In one line: It gives you grids of exact fractions, so array math avoids rounding entirely.

## What it gives you

When you need a whole grid of values to stay exact, decimals will not do; they round. This provides a grid where every cell is a precise fraction, a top number over a bottom number, kept compact for speed. Each cell is automatically reduced to lowest terms with a tidy, consistent sign, so equal fractions look the same and comparisons behave. It converts cleanly to and from the system's general grid form, staying fully compatible with the rest of the stack. If the count of values does not match the shape you declared, it refuses rather than proceed with mismatched data.

## Why you will be glad

- Whole grids of fractions stay perfectly exact, with no creeping rounding.
- Every cell is reduced and sign-normalized, so the data is clean and comparable.
- Shape mismatches are caught immediately instead of corrupting results silently.

## Where it fits

This is the exact-fraction specialization of the SIM tensor stack. It registers against the shared grid interface as the rational backend, giving the constellation a way to hold arrays where exactness matters more than the convenience of decimals.
