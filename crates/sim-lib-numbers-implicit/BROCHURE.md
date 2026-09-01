# sim-lib-numbers-implicit

In one line: Evidence-carrying Radau IIA integration for stiff ODEs and index-1 DAEs.

## What it gives you

`sim-lib-numbers-implicit` advances stiff ODEs and genuine index-1 DAEs with three-stage, fifth-order Radau IIA. Every result explains where its Jacobian came from, why it was rebuilt, how often its factor was reused, whether Newton failed, and what rank and conditioning the accepted linear systems exhibited. Collocation polynomials provide continuous output and event location. Dense output is never manufactured from an unconverged stage, while singular Jacobians and work exhaustion are explicit refusals rather than dubious values. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-implicit owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
