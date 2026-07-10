# I64 tensor arithmetic (descriptor)

Checked scalar-add over an i64 tensor is tensor arithmetic, which the sandbox eval stack does
not execute. The runnable `numbers/tensor` recipe constructs an i64 tensor live; this recipe
documents the checked scalar-add surface.
