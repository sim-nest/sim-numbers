# Execute a tensor expression

Build a length-4 `i64` vector and reshape it into a `2x2` tensor. The same
expression is valid through the tensor execution site because the site binds the
active tensor executor in a child environment and evaluates ordinary tensor
operations without changing the caller environment.
