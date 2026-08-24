# Root finding that shows its work

`sim-lib-numbers-root` does not collapse every plausible answer into a bare
number. Bisection and Brent-Dekker return a sign-changing bracket; safeguarded
Newton and secant return a residual-certified estimate. Vector Newton reports
SVD-derived rank and every damped step, while bounded Broyden reports exactly
when it refreshed its Jacobian.

- Invalid brackets are rejected before iteration.
- Analytic, automatic, and finite-difference derivatives remain distinguishable.
- Non-finite values, discontinuities, flat derivatives, cycling, stagnation,
  rank loss, and exhausted budgets are separate outcomes.
- Execution identity and work counts make deterministic replay reviewable.
