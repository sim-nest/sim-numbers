# sim-lib-numbers-extended

In one line: Normalized double-double scalar and runtime number domain for SIM.

## What it gives you

`sim-lib-numbers-extended` injects a normalized double-double scalar through SIM's shared numerical algorithm contract. It retains about 106 significant binary digits across cancellation-heavy work, preserves exact component bits in runtime records, and participates in the canonical promotion lattice. Use it when binary64 is almost sufficient and an arbitrary-precision engine would obscure bounded numerical work. The checked convergence specimen runs root, quadrature, ODE, polynomial, and decomposition kernels over both scalar types through the same generic implementations. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-extended owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
