# sim-lib-numbers-tensor-decomp

Bounded Householder QR, symmetric eigen, balanced-Hessenberg real Schur, and
one-sided Jacobi SVD with independently checked reconstruction and orthogonality
certificates. SVD-derived rank, condition, pseudoinverse, least-squares, and
null-space operations share one caller-declared cutoff. Provider factors are
accepted only after local shape, ordering, residual, and identity checks.
