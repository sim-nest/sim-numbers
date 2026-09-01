# sim-lib-numbers-method

In one line: Validated bounded numerical method plans and canonical cross-domain evidence.

## What it gives you

`sim-lib-numbers-method` gives every numerical algorithm the same durable answers to the questions that matter: which registered method ran, what bound admitted it, why it stopped, which exact criterion converged, what error it achieved, at what precision, and under which execution identity. Success cannot be forged from a boolean. Convergence must name a requested criterion and carry an achieved value within its threshold. Work counters saturate safely. Canonical `Datum` projections preserve every finite binary64 bit--including signed zero--and reject ambiguous NaNs. Existing dense-solve, statistics, and signal reports remain the authoritative domain evidence. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-method owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
