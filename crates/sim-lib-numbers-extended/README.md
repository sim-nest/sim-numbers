# sim-lib-numbers-extended

SIM's finite extended-precision scalar domain. `DoubleDouble` stores the exact
bits of a normalized unevaluated sum `hi + lo`, supplies the `RealScalar`
algorithm bridge, and is registered as the single `numbers/extended` runtime
`NumberDomain`.

The arithmetic uses error-free binary64 transforms. Elementary functions use
double-double argument reduction and correction iterations; the crate makes
tested accuracy claims, not a correct-rounding claim.

The checked compact-domain specimens bound square-root residuals below
`1e-30`, exponential/logarithm round trips below `2e-30`, and the trigonometric
identity residual below `2e-30`. These are tested bounds for the documented
specimens, not global error proofs.
