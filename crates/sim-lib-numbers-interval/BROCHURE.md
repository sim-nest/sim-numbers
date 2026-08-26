# sim-lib-numbers-interval

In one line: Sealed, evidence-carrying interval certification for SIM.

## What it gives you

Make numerical proof a type boundary. `EstimateInterval` reports useful uncertainty without claiming containment. `CertifiedInterval` is issued only by reviewed kernels, carries replayable evidence, and is the only input accepted by definite threshold classification and verified root tests. Directed binary64 operations cover every exceptional case explicitly; exact rational intervals provide an independent oracle. Unsupported elementary functions refuse certification instead of dressing a library approximation as proof. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-interval owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
