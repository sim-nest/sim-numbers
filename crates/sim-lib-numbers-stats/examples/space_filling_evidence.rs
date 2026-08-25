use sim_lib_numbers_stats::{LatinHypercubePlan, Scramble, SobolPlan, SweepPlan, UntestedRegion};

fn main() {
    let latin = LatinHypercubePlan {
        dimensions: 3,
        points: 8,
        seed: 2026,
        max_work: 21,
        untested_regions: vec![],
    }
    .generate()
    .expect("bounded Latin design");
    let sobol = SobolPlan {
        dimensions: 2,
        points: 8,
        skip: 4,
        scramble: Scramble::DigitalShift,
        seed: 2026,
        max_work: 16,
        untested_regions: vec![],
    }
    .generate()
    .expect("reviewed Sobol design");
    let covered = SweepPlan {
        inject_lower_boundary: true,
        inject_upper_boundary: true,
        untested_regions: vec![UntestedRegion {
            label: "unstable-corner".into(),
            reason: "excluded by study protocol".into(),
        }],
    }
    .apply(sobol);
    println!(
        "latin={} occupancy={:?} sampler={:?}",
        latin.points.len(),
        latin.coverage.stratum_occupancy,
        latin.coverage.sampler
    );
    println!(
        "sobol={} boundaries={:?} duplicates={:?} exclusions={:?}",
        covered.points.len(),
        covered.coverage.boundary_injections,
        covered.coverage.duplicates,
        covered.coverage.untested_regions
    );
}
