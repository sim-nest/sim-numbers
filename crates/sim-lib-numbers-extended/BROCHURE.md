# Extended precision without duplicated solvers

`sim-lib-numbers-extended` injects a normalized double-double scalar through
SIM's shared numerical algorithm contract. It retains about 106 significant
binary digits across cancellation-heavy work, preserves exact component bits
in runtime records, and participates in the canonical promotion lattice.

Use it when binary64 is almost sufficient and an arbitrary-precision engine
would obscure bounded numerical work. The checked convergence specimen runs
root, quadrature, ODE, polynomial, and decomposition kernels over both scalar
types through the same generic implementations.
