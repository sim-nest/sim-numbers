# sim-lib-numbers-method

Validated bounded numerical method plans and common evidence for SIM.

The crate supplies criterion-specific tolerances, saturating work receipts,
non-forgeable termination evidence, exact canonical `Datum` projections, and
adapters that leave dense-solve, statistics, and signal reports intact. It is
not a runtime number domain or a promotion lattice. Algorithms use the sealed
`sim_lib_numbers_core::RealScalar` bridge, initially implemented by `f64`.

```rust
use sim_lib_numbers_method::{CriterionId, ErrorMeasure, MethodId, MethodPlan,
    ToleranceSet, WorkLimit};

let plan = MethodPlan::new(
    MethodId::new(MethodId::DENSE_SCALED_PIVOT)?,
    WorkLimit::new(10_000)?,
    ToleranceSet::new([ErrorMeasure::new(CriterionId::ResidualNorm, 1e-10)?])?,
);
# Ok::<(), Box<dyn std::error::Error>>(())
```

See the embedded `numbers/method` cookbook for inspection and adapter guidance.
