# Inspect bounded numerical evidence

Construct a `MethodPlan` before execution, retain the domain report, and project
it through the matching adapter. Inspect `method`, `termination`, `work`,
`requested`, `achieved`, `precision`, and `execution` from the canonical Datum.
Never infer convergence from a domain boolean alone: the common constructor
requires a named requested criterion and an achieved measure within threshold.
