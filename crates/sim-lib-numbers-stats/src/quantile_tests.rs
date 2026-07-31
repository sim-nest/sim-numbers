use super::quantile::*;

// conformance: exact and hard-bounded mergeable streaming quantiles.

fn policy() -> QuantilePolicy {
    QuantilePolicy::new(0.02, 32, 512).unwrap()
}

#[test]
fn small_data_uses_the_exact_interpolated_reference() {
    let mut sketch = QuantileSketch::new(policy()).unwrap();
    for value in [9.0, 1.0, 5.0, 3.0] {
        sketch.insert(value).unwrap();
    }
    let estimate = sketch.estimate(0.5).unwrap();
    assert_eq!(
        estimate.value,
        exact_quantile(&[9.0, 1.0, 5.0, 3.0], 0.5).unwrap()
    );
    assert!(estimate.exact);
    assert_eq!(estimate.retained_entries, 4);
}

#[test]
fn large_stream_is_bounded_and_tracks_requested_ranks() {
    let mut sketch = QuantileSketch::new(policy()).unwrap();
    for value in 0..10_000 {
        sketch.insert(value as f64).unwrap();
    }
    assert!(sketch.retained_entries() <= policy().max_summary_entries);
    assert!(sketch.retained_entry_bytes() > 0);
    for quantile in [0.0, 0.1, 0.5, 0.9, 1.0] {
        let estimate = sketch.estimate(quantile).unwrap();
        let actual_rank = estimate.value / 9_999.0;
        assert!((actual_rank - quantile).abs() <= policy().rank_error + 0.001);
        assert!(estimate.rank_lower <= actual_rank);
        assert!(actual_rank <= estimate.rank_upper);
        assert!(!estimate.exact);
    }
}

#[test]
fn independently_built_summaries_merge_without_source_replay() {
    let mut whole = QuantileSketch::new(policy()).unwrap();
    let mut left = QuantileSketch::new(policy()).unwrap();
    let mut right = QuantileSketch::new(policy()).unwrap();
    for value in 0..4_000 {
        whole.insert(value as f64).unwrap();
        if value % 2 == 0 {
            left.insert(value as f64).unwrap();
        } else {
            right.insert(value as f64).unwrap();
        }
    }
    left.merge(&right).unwrap();
    assert_eq!(left.len(), whole.len());
    assert!(left.retained_entries() <= policy().max_summary_entries);
    for quantile in [0.1, 0.5, 0.9] {
        let merged = left.estimate(quantile).unwrap().value;
        assert!((merged / 3_999.0 - quantile).abs() <= 2.0 * policy().rank_error + 0.002);
    }
}

#[test]
fn invalid_values_policies_and_memory_fail_closed() {
    assert!(matches!(
        QuantilePolicy::new(0.5, 1, 1),
        Err(QuantileError::InvalidPolicy { .. })
    ));
    assert!(matches!(
        exact_quantile(&[1.0, f64::NAN], 0.5),
        Err(QuantileError::NonFinite { .. })
    ));

    let tight = QuantilePolicy::new(0.0, 2, 2).unwrap();
    let mut sketch = QuantileSketch::new(tight).unwrap();
    sketch.insert(1.0).unwrap();
    sketch.insert(2.0).unwrap();
    let before = sketch.clone();
    assert!(matches!(
        sketch.insert(3.0),
        Err(QuantileError::MemoryLimit { .. })
    ));
    assert_eq!(sketch, before);
}
