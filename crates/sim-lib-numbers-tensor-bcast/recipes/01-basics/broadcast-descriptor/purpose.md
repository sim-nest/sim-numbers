# Tensor broadcasting (descriptor)

Broadcasting aligns a scalar or lower-rank tensor against a higher-rank shape before an
elementwise op. It is tensor arithmetic, which the sandbox eval stack does not execute
(it constructs tensors but does not operate on them). The runnable `numbers/tensor`,
`tensor-cmplxf`, and `tensor-rat64` recipes show live tensor construction.
