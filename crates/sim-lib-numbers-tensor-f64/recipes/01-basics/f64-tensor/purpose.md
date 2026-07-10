# F64 tensor arithmetic (descriptor)

Adding a scalar to an f64 tensor is tensor arithmetic, which the sandbox eval stack does not
execute (it constructs tensors but does not operate on them). The runnable `numbers/tensor`
recipe constructs a 2x2 i64 tensor live; this recipe documents the f64 scalar-add surface.
