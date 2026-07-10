# Boolean promotes to integer

Multiply the boolean literal `true` by `6`. `bool` is the base of the
number-promotion lattice, so `true` widens to `1` and the product computes `6` --
a real arithmetic result across a domain edge, not a quoted descriptor.
