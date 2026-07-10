# Fixed-width integer domains (descriptor)

The `numbers/fixed` domains (`i8`..`i128`, `u8`..`u128`) provide checked fixed-width
integer arithmetic with explicit scale. Their typed literals are not yet exposed through
the sandbox reader, so this recipe documents the fixed-width domain and its checked-arith
surface. Live integer arithmetic is shown by the runnable `numbers/i64` recipe.
