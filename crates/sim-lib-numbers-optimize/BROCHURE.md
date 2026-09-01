# sim-lib-numbers-optimize

In one line: Bounded optimization and evidence-carrying least squares for SIM.

## What it gives you

Minimize scalar and smooth multivariate objectives, fit nonlinear models, and solve bounded linear least squares without hiding bounds behind clipping. SIM returns brackets, active sets, rank, residuals, covariance availability, termination, and resource use. Optional assignment conversion delegates to the single certified discrete solver and preserves its certificate and work receipt. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-optimize owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
