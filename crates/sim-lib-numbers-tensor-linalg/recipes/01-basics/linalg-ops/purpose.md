# Tensor executor math

Dot product, matrix multiply, whole-tensor reductions, Euclidean norm, and f32/f64
transcendentals use the `numbers/tensor-linalg` callable surface. Reducible and
matrix operations submit checked tensor executor requests before the CPU fallback
answers, so the same call sites can run on a placed tensor site.
