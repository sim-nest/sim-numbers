# sim-lib-numbers-tensor-decomp

Factor matrices without surrendering the evidence. QR supports optional column
pivoting and numerical-rank reporting; symmetric eigen refuses unapproved
asymmetry and returns ordered values with optional vectors. Both preserve their
inputs, honor hard work and dimension limits, and certify their results.

General real spectra use balanced Hessenberg reduction and bounded paired-shift
Schur iteration, including canonical complex conjugate eigenvalues. Stable
one-sided Jacobi SVD handles rectangular and rank-deficient matrices without
forming normal equations. One explicit cutoff governs rank, two-norm condition,
pseudoinverse, least squares, and null spaces. Exhausted factors are either
rejected or explicitly partial and can never enter derived solves as success.
