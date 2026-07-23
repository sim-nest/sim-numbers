# Construct a rational-cell tensor

Build a length-2 tensor typed over the `numbers/rational` element domain, with exact
fraction cells `1/2` and `2/3`. The dimension is encoded with `expr:number` so
it remains a citizen integer while the cells use the loaded rational number
domain. Read-construct builds the shaped, exactly-typed value directly -- a live
tensor of rationals, not a quoted descriptor.
