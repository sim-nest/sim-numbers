# Bitwise tensor ops (descriptor)

Packed-boolean tensors support bitwise `and`/`or`/`xor`. Those are tensor operations, which
the sandbox eval stack does not execute -- it constructs tensors but does not operate on
them. The runnable tensor-construction recipes demonstrate the live construction side.
