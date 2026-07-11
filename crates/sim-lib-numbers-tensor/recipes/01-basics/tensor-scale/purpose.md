# Scale a tensor by a scalar

Build the length-2 `i64` vector `(vec 3 4)` and multiply it by the scalar `2`. The tensor
domain broadcasts the scalar across every cell and computes a real shaped result `[6 8]` --
a live elementwise product, not a quoted shape descriptor.
