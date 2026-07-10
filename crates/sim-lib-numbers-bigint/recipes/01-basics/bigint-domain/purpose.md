# Overflow promotes into bigint

Raise `2` to the `64`th power. The result exceeds the 64-bit integer range, so the
promotion lattice widens it into the arbitrary-precision `bigint` domain, which
computes the exact `18446744073709551616` -- no wraparound, no loss.
