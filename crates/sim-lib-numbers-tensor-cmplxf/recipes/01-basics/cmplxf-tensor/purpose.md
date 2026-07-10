# Construct a complex-cell tensor

Build a length-2 tensor whose cells are themselves `#(numbers/Complex ...)` values, typed
over the `numbers/complex` element domain. Read-construct assembles both the outer tensor
and its inner complex cells into one real typed value -- a live demonstration of the
tensor dtype system, not a quoted shape descriptor.
