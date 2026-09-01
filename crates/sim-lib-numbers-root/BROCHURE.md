# sim-lib-numbers-root

In one line: Bracketed and residual-carrying scalar and vector root finding for SIM.

## What it gives you

`sim-lib-numbers-root` does not collapse every plausible answer into a bare number. Bisection and Brent-Dekker return a sign-changing bracket; safeguarded Newton and secant return a residual-certified estimate. Vector Newton reports SVD-derived rank and every damped step, while bounded Broyden reports exactly when it refreshed its Jacobian. Invalid brackets are rejected before iteration. Analytic, automatic, and finite-difference derivatives remain distinguishable. Non-finite values, discontinuities, flat derivatives, cycling, stagnation, rank loss, and exhausted budgets are separate outcomes. Execution identity and work counts make deterministic replay reviewable. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-root owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
