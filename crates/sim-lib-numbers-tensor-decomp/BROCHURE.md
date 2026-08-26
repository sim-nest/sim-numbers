# sim-lib-numbers-tensor-decomp

In one line: Bounded, evidence-carrying dense matrix decompositions for SIM tensors.

## What it gives you

Factor matrices without surrendering the evidence. QR supports optional column pivoting and numerical-rank reporting; symmetric eigen refuses unapproved asymmetry and returns ordered values with optional vectors. Both preserve their inputs, honor hard work and dimension limits, and certify their results. General real spectra use balanced Hessenberg reduction and bounded paired-shift Schur iteration, including canonical complex conjugate eigenvalues. Stable one-sided Jacobi SVD handles rectangular and rank-deficient matrices without forming normal equations. One explicit cutoff governs rank, two-norm condition, pseudoinverse, least squares, and null spaces. Exhausted factors are either rejected or explicitly. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-tensor-decomp owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
