# Simplify a symbolic CAS product

Simplify the symbolic product `#(numbers/Cas v1 (* 0 x))` with `cas/simplify`. The CAS
domain applies real algebraic rewriting -- a product with a zero factor absorbs to `0` --
computing a live simplified result, not a quoted descriptor.
