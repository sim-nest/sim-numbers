# sim-lib-numbers-tensor-f32

In one line: It gives you compact single-precision grids for data and model math that do not need f64 storage.

## What it gives you

Many numeric workloads want the shape discipline of tensors without the memory footprint of double precision. This provides a tensor backend that stores each cell as native `f32`, keeps the same canonical tensor interface as the rest of the stack, and can still round-trip through the uniform representation when a generic runtime component needs to inspect it.

## Why you will be glad

- Single-precision arrays take half the storage of f64 arrays.
- The backend plugs into the same tensor descriptor surface as the existing typed stores.
- Shape mismatches fail before a malformed tensor can enter a calculation.

## Where it fits

This is the single-precision storage backend for the SIM tensor stack. It sits beside the f64, i64, rational, complex, and bit tensor backends and gives GPU-oriented or memory-sensitive code an ordinary typed tensor component.
