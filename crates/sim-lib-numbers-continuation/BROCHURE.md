# sim-lib-numbers-continuation

In one line: Bounded pseudo-arclength continuation with reviewable predictor/corrector evidence.

## What it gives you

`sim-lib-numbers-continuation` follows a residual manifold through turning points using secant/tangent prediction and a bordered Newton corrector. Every accepted point carries its residual, tangent, normal, corrector report, and fold classification; rejected predictions and adaptive step decisions remain reviewable. Work, domain, and loop limits are explicit, so traces are bounded and a recorded pair of points can resume deterministically. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-continuation owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
