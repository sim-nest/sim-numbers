use std::sync::Arc;

use sim_kernel::{AssocTable, Cx, DefaultFactory, EagerPolicy, Symbol};
use sim_lib_numbers_signal::{
    PlacementPolicy, TensorView, TransformKind, TransformPlan, TransformResources,
    read_blocked_tensor, transform_nd, transform_nd_blocked, write_blocked_tensor,
};

fn main() {
    let cells = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let view = TensorView::complex(&cells, vec![2, 2], vec![2, 1], 0).unwrap();
    let resources = TransformResources {
        max_scratch_bytes: 1024,
        block_len: 2,
    };
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let store = AssocTable::new();
    let blocked = write_blocked_tensor(
        &mut cx,
        &store,
        Symbol::qualified("recipe", "blocked-transform"),
        &view,
        resources,
    )
    .unwrap();

    let mut blocked_plan = TransformPlan::new(TransformKind::Fft, 1);
    blocked_plan.placement = PlacementPolicy::InPlace;
    let report =
        transform_nd_blocked(&mut cx, &store, &blocked, &[0, 1], &blocked_plan, resources)
            .unwrap();
    let blocked_output = read_blocked_tensor(&mut cx, &store, &blocked).unwrap();

    let in_memory_plan = TransformPlan::new(TransformKind::Fft, 1);
    let in_memory = transform_nd(view, &[0, 1], &in_memory_plan).unwrap();
    println!("shape={:?}", blocked.shape());
    println!(
        "passes={} scratch={} bounded={}",
        report.passes,
        report.scratch_bytes,
        report.scratch_bytes <= resources.max_scratch_bytes
    );
    println!("blocked-equals-memory={}", blocked_output == in_memory.output);
}
