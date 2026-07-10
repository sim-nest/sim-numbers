# Statistics helpers (descriptor)

This records the `numbers/stats` helper surface: Bayesian update, entropy, mean, variance,
sample-variance, and the fairness ratios. These operate over the f64 domain through library
functions outside the sandbox eval stack, so the recipe documents the surface rather than
computing a single value live.
