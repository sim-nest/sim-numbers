# Compose arithmetic operations

Nest `math/mul` inside `math/sub` to evaluate `(6 * 7) - 2`. The cross-domain
arithmetic entry points compose into a single expression that computes `40` --
real evaluation, not a quoted operator list.
