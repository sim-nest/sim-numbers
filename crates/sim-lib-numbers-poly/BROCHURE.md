# Certified coefficient algebra

`sim-lib-numbers-poly` provides ordinary, Laurent, and Puiseux coefficient
records without confusing their exponent domains. It evaluates by Horner with
error bounds, performs exact rational division and GCD, recovers complex roots
with multiplicity and backward-error evidence, and constructs Pade
approximants only when their defining system is numerically full rank.
