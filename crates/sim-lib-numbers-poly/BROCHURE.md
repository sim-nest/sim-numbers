# sim-lib-numbers-poly

In one line: Evidence-carrying dense coefficient polynomials for SIM.

## What it gives you

`sim-lib-numbers-poly` provides ordinary, Laurent, and Puiseux coefficient records without confusing their exponent domains. It evaluates by Horner with error bounds, performs exact rational division and GCD, recovers complex roots with multiplicity and backward-error evidence, and constructs Pade approximants only when their defining system is numerically full rank. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-poly owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
