# Automatic differentiation (descriptor)

Automatic differentiation evaluates a function over dual numbers, carrying a derivative
tape alongside each value. That tape machinery runs outside the cookbook sandbox eval
stack (which supports `math/*` and read-construct, not tape propagation), so this recipe
documents the `numbers/ad` dual-tape interface rather than computing a gradient live.
