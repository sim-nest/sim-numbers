# Quantities that know what they mean

`sim-lib-numbers-quantity` keeps magnitude, dimension, semantic kind, unit, and
measurement role separate. Exact unit conversions stay in the installed scalar
domain, affine temperature points obey affine laws, and energy cannot be
silently accepted as torque merely because both have the same SI exponents.

Use it for durable scientific APIs, checked physical data exchange, and runtime
values that must survive Shape admission and general expression codecs without
growing the SIM kernel.
