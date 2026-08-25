# sim-lib-numbers-quantity

Semantic physical quantities for SIM. This crate composes installed scalar
domains with exact rational SI-dimension vectors, open content-identified
measure kinds, exact affine units, point/interval roles, Shape constraints, and
a canonical runtime read constructor.

The scalar is never reduced to `f64` during conversion. Energy and torque are
the canonical example of equal dimensions with intentionally incompatible
semantic kinds.

```rust
use sim_lib_numbers_quantity::{ExactScalar, MeasureRole, Quantity, JOULE, energy_kind};

let energy = Quantity::new(
    ExactScalar::from(12),
    energy_kind().dimension().clone(),
    Some(energy_kind()),
    Some(JOULE()),
    MeasureRole::Interval,
)?;
assert_eq!(energy.scalar(), &ExactScalar::from(12));
# Ok::<(), Box<dyn std::error::Error>>(())
```
