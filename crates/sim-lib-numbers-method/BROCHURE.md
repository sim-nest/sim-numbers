# Numerical methods that can show their work

`sim-lib-numbers-method` gives every numerical algorithm the same durable
answers to the questions that matter: which registered method ran, what bound
admitted it, why it stopped, which exact criterion converged, what error it
achieved, at what precision, and under which execution identity.

Success cannot be forged from a boolean. Convergence must name a requested
criterion and carry an achieved value within its threshold. Work counters
saturate safely. Canonical `Datum` projections preserve every finite binary64
bit—including signed zero—and reject ambiguous NaNs. Existing dense-solve,
statistics, and signal reports remain the authoritative domain evidence while
small adapters make their shared facts uniformly inspectable.
