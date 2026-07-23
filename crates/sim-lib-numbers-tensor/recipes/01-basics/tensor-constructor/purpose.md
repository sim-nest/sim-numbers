# Construct a 2x2 integer tensor

Build a shaped `2x2` `i64` tensor from its reader form
`#(numbers/Tensor v1 ((expr:number citizen/int "2") (expr:number citizen/int "2")) (1 2 3 4) numbers/i64)`.
The explicit dimension escapes keep the read-constructor shape fields in the
citizen integer domain even when `numbers/i64` is loaded for tensor cells. The
tensor domain constructs a real typed n-dimensional value through read-construct
and renders its row structure as `((1 2) (3 4))` -- a live shaped value, not a
quoted shape descriptor.
