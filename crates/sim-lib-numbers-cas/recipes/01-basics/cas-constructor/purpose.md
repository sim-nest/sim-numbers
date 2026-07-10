# Build a symbolic CAS term

Construct the CAS expression `x + 1` as `#(numbers/Cas v1 (+ x 1))`. The CAS domain builds
and normalizes it to the symbolic term `(+ 1 x)` -- a real symbolic value, not a quote.
