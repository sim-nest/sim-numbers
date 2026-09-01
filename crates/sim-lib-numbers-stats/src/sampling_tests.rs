use super::*;

#[test]
fn sampler_replays_refuses_and_forks_independently() {
    let mut sampler = SeededSampler::with_max_words(7, 2);
    let first = sampler.try_next_u64().unwrap();
    let receipt = sampler.receipt();
    let second = sampler.try_next_u64().unwrap();
    assert_eq!(
        SeededSampler::replay(receipt).try_next_u64().unwrap(),
        second
    );
    assert!(matches!(
        sampler.try_next_u64(),
        Err(DesignError::WorkLimit {
            required: 3,
            limit: 2
        })
    ));
    let root = SeededSampler::new(7);
    let mut left = root.fork(1);
    let mut right = root.fork(2);
    assert_ne!(left.try_next_u64().unwrap(), right.try_next_u64().unwrap());
    assert_eq!(first, 7_191_089_600_892_374_487);
}

#[test]
fn sobol_prefix_and_latin_occupancy_are_canonical() {
    let sobol = SobolPlan {
        dimensions: 2,
        points: 4,
        skip: 0,
        scramble: Scramble::None,
        seed: 0,
        max_work: 8,
        untested_regions: vec![],
    }
    .generate()
    .unwrap();
    assert_eq!(
        sobol.points,
        vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.25, 0.75]
        ]
    );
    assert!(matches!(
        SobolPlan {
            dimensions: 5,
            points: 1,
            skip: 0,
            scramble: Scramble::None,
            seed: 0,
            max_work: 5,
            untested_regions: vec![]
        }
        .generate(),
        Err(DesignError::UnsupportedDimension { maximum: 4, .. })
    ));
    let latin = LatinHypercubePlan {
        dimensions: 3,
        points: 8,
        seed: 11,
        max_work: 21,
        untested_regions: vec![],
    }
    .generate()
    .unwrap();
    assert_eq!(latin.coverage.stratum_occupancy, vec![vec![1; 8]; 3]);
    assert!(matches!(
        LatinHypercubePlan {
            dimensions: 2,
            points: 8,
            seed: 1,
            max_work: 13,
            untested_regions: vec![]
        }
        .generate(),
        Err(DesignError::WorkLimit {
            required: 14,
            limit: 13
        })
    ));
}

#[test]
fn coverage_and_distribution_edges_are_explicit() {
    let base = SobolPlan {
        dimensions: 1,
        points: 2,
        skip: 0,
        scramble: Scramble::None,
        seed: 0,
        max_work: 2,
        untested_regions: vec![],
    }
    .generate()
    .unwrap();
    let swept = SweepPlan {
        inject_lower_boundary: true,
        inject_upper_boundary: true,
        untested_regions: vec![UntestedRegion {
            label: "singularity".into(),
            reason: "caller excluded".into(),
        }],
    }
    .apply(base);
    assert_eq!(swept.coverage.boundary_injections, vec![0, 3]);
    assert_eq!(swept.coverage.duplicates, vec![(1, 0)]);
    assert_eq!(swept.coverage.untested_regions.len(), 1);

    let moments =
        standardized_moments(&[-1.0, 0.0, 1.0, 2.0], MomentConvention::Population).unwrap();
    assert_eq!(moments.skewness, 0.0);
    let same =
        kolmogorov_smirnov_two_sample(&[0.0, 1.0], &[0.0, 1.0], KsMethod::ExactStatistic).unwrap();
    assert_eq!(same.statistic, 0.0);
    assert_eq!(same.p_value, None);
    let separated =
        kolmogorov_smirnov_two_sample(&[0.0, 1.0], &[2.0, 3.0], KsMethod::Asymptotic).unwrap();
    assert_eq!(separated.statistic, 1.0);
    assert!(separated.p_value.unwrap() <= 1.0);
    assert!(kolmogorov_smirnov_one_sample(&[0.5], |_| 2.0, KsMethod::ExactStatistic).is_err());
}
