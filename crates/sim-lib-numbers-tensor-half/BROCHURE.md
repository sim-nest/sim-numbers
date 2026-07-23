# sim-lib-numbers-tensor-half

In one line: It gives you compact half-precision tensor storage with explicit f32 widening for CPU work.

## What it gives you

Model and GPU-oriented data often lives in `f16` or `bf16` buffers. This provides those two storage backends as ordinary SIM tensor descriptors while keeping CPU arithmetic honest: when a host-side helper computes over the compact cells, it widens them to `f32` first and returns an `f32` tensor or value.

## Why you will be glad

- Half-precision tensors fit the storage shape used by accelerators and model weights.
- CPU arithmetic widens to f32 instead of hiding precision and overflow behavior.
- Both f16 and bf16 plug into the same tensor discovery surface as the other typed backends.

## Where it fits

This is the compact floating-point storage backend for the SIM tensor stack. Explicit tensor casts move values into and out of the half domains, while scalar math remains in the established number tower.
