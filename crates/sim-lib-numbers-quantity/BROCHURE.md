# sim-lib-numbers-quantity

In one line: Exact semantic dimensions, units, and generic physical quantities for SIM.

## What it gives you

`sim-lib-numbers-quantity` keeps magnitude, dimension, semantic kind, unit, and measurement role separate. Exact unit conversions stay in the installed scalar domain, affine temperature points obey affine laws, and energy cannot be silently accepted as torque merely because both have the same SI exponents. Use it for durable scientific APIs, checked physical data exchange, and runtime values that must survive Shape admission and general expression codecs without growing the SIM kernel. The contract keeps inputs, outputs, limits, and refusal cases explicit, so callers can compose the capability without acquiring unrelated host, transport, or product authority. Stable records make the result suitable for tests, inspection, and deterministic integration.

## Why you will be glad

- The public contract makes supported behavior, limits, and typed failures visible before integration.
- One owning crate prevents neighboring libraries from growing competing copies of the same policy.
- Deterministic records and checked tests keep adapters reviewable when implementations evolve.

## Where it fits

Within SIM, sim-lib-numbers-quantity owns only the focused contract described above. Adjacent runtime libraries, platform adapters, codecs, and user surfaces can build around it while retaining their own policy. That boundary keeps the kernel small, avoids competing implementations, and lets this capability evolve without forcing unrelated components to change.
