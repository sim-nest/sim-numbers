use super::{clustering::*, gmm::*};

// conformance: generated, degenerate, permuted, and bounded clustering fixtures.

fn generated_clusters() -> Vec<Vec<f64>> {
    let centers = [[-5.0, -1.0], [0.0, 5.0], [6.0, 0.5]];
    centers
        .iter()
        .flat_map(|center| {
            (0..12).map(move |index| {
                let x_offset = (index % 4) as f64 * 0.08 - 0.12;
                let y_offset = (index / 4) as f64 * 0.09 - 0.09;
                vec![center[0] + x_offset, center[1] + y_offset]
            })
        })
        .collect()
}

fn close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "expected {left} to be within {tolerance} of {right}"
    );
}

#[test]
fn seeded_kmeans_restarts_are_deterministic_and_select_lowest_inertia() {
    let points = generated_clusters();
    let control = KMeansControl::new(4, 50, 1.0e-10, 20_000, 4).unwrap();
    let first = fit_kmeans(&points, 3, control).unwrap();
    let second = fit_kmeans(&points, 3, control).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.restarts.len(), 4);
    assert_eq!(first.termination, KMeansSearchTermination::Completed);
    assert!(first.work <= control.max_work);
    let selected = &first.restarts[first.selected_restart];
    assert!(
        first
            .restarts
            .iter()
            .all(|candidate| selected.inertia <= candidate.inertia)
    );
    assert!(first.restarts.iter().all(|candidate| candidate.converged));
    for (centroid, expected) in
        first
            .model
            .centroids
            .iter()
            .zip([[-5.0, -1.0], [0.0, 5.0], [6.0, 0.5]])
    {
        close(centroid[0], expected[0], 0.15);
        close(centroid[1], expected[1], 0.15);
    }
}

#[test]
fn empty_clusters_are_repaired_and_point_permutation_preserves_the_solution() {
    let degenerate = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
    ];
    let repaired = fit_kmeans(
        &degenerate,
        3,
        KMeansControl::new(9, 20, 0.0, 5_000, 2).unwrap(),
    )
    .unwrap();
    assert!(
        repaired
            .restarts
            .iter()
            .any(|evidence| evidence.empty_cluster_repairs > 0)
    );
    assert!(
        repaired
            .restarts
            .iter()
            .all(|evidence| evidence.inertia.is_finite())
    );

    let points = generated_clusters();
    let mut permuted = points.clone();
    permuted.reverse();
    let control = KMeansControl::new(23, 50, 1.0e-10, 20_000, 4).unwrap();
    let original = fit_kmeans(&points, 3, control).unwrap();
    let reversed = fit_kmeans(&permuted, 3, control).unwrap();
    assert_eq!(
        original.model.centroids.len(),
        reversed.model.centroids.len()
    );
    for (left, right) in original
        .model
        .centroids
        .iter()
        .zip(&reversed.model.centroids)
    {
        close(left[0], right[0], 1.0e-12);
        close(left[1], right[1], 1.0e-12);
    }
    close(
        original.restarts[original.selected_restart].inertia,
        reversed.restarts[reversed.selected_restart].inertia,
        1.0e-12,
    );
}

#[test]
fn diagonal_gmm_is_seeded_log_domain_and_model_selection_is_explicit() {
    let points = generated_clusters();
    let spec = GmmSpec::new(
        3,
        CovarianceType::Diagonal,
        1.0e-6,
        SingularComponentPolicy::default(),
    )
    .unwrap();
    let control = GmmControl::new(4, 80, 1.0e-9, 100_000).unwrap();
    let first = fit_gmm(&points, spec, control).unwrap();
    let second = fit_gmm(&points, spec, control).unwrap();

    assert_eq!(first, second);
    assert!(first.evidence.work <= control.max_work);
    assert!(first.evidence.log_likelihood.is_finite());
    assert!(
        first
            .evidence
            .likelihood_history
            .windows(2)
            .all(|pair| pair[1] + 1.0e-8 >= pair[0])
    );
    assert_eq!(first.evidence.model_selection.parameters, 14);
    assert!(first.evidence.model_selection.aic.is_finite());
    assert!(first.evidence.model_selection.bic.is_finite());
    let responsibilities = first.model.responsibilities(&points).unwrap();
    assert!(responsibilities.iter().all(|row| {
        row.iter().all(|value| value.is_finite()) && (row.iter().sum::<f64>() - 1.0).abs() < 1.0e-12
    }));
    assert_eq!(first.model.predict(&points).unwrap().len(), points.len());
}

#[test]
fn full_covariance_and_component_permutation_retain_likelihood() {
    let points = (0..40)
        .flat_map(|index| {
            let x = index as f64 * 0.04;
            [vec![-3.0 + x, -2.0 + 0.7 * x], vec![4.0 + x, 3.0 + 0.5 * x]]
        })
        .collect::<Vec<_>>();
    let spec = GmmSpec::new(
        2,
        CovarianceType::Full,
        1.0e-5,
        SingularComponentPolicy::default(),
    )
    .unwrap();
    let report = fit_gmm(
        &points,
        spec,
        GmmControl::new(31, 80, 1.0e-9, 500_000).unwrap(),
    )
    .unwrap();
    assert_eq!(report.evidence.model_selection.parameters, 11);
    for covariance in &report.model.covariances {
        let GaussianCovariance::Full(matrix) = covariance else {
            panic!("expected full covariance");
        };
        assert!(matrix[0][0] >= spec.regularization);
        assert!(matrix[1][1] >= spec.regularization);
        close(matrix[0][1], matrix[1][0], 1.0e-14);
    }

    let original_likelihood = report.model.log_likelihood(&points).unwrap();
    let mut permuted = report.model.clone();
    permuted.weights.reverse();
    permuted.means.reverse();
    permuted.covariances.reverse();
    close(
        original_likelihood,
        permuted.log_likelihood(&points).unwrap(),
        1.0e-10,
    );
}

#[test]
fn singular_policy_and_non_finite_or_unadmitted_inputs_fail_closed() {
    let degenerate = vec![vec![2.0, 2.0]; 6];
    let repair_spec = GmmSpec::new(
        3,
        CovarianceType::Diagonal,
        1.0e-6,
        SingularComponentPolicy::Reinitialize {
            minimum_weight: 0.4,
        },
    )
    .unwrap();
    let repaired = fit_gmm(
        &degenerate,
        repair_spec,
        GmmControl::new(2, 5, 0.0, 20_000).unwrap(),
    )
    .unwrap();
    assert!(repaired.evidence.singular_component_repairs > 0);
    assert!(repaired.evidence.log_likelihood.is_finite());

    let fail_spec = GmmSpec::new(
        3,
        CovarianceType::Diagonal,
        1.0e-6,
        SingularComponentPolicy::Fail {
            minimum_weight: 0.4,
        },
    )
    .unwrap();
    assert!(matches!(
        fit_gmm(
            &degenerate,
            fail_spec,
            GmmControl::new(2, 5, 0.0, 20_000).unwrap()
        ),
        Err(ClusteringError::SingularComponent { .. })
    ));

    let non_finite = vec![vec![0.0, f64::NAN], vec![1.0, 1.0]];
    assert!(matches!(
        fit_kmeans(&non_finite, 1, KMeansControl::default()),
        Err(ClusteringError::NonFiniteInput { .. })
    ));
    assert!(matches!(
        fit_gmm(&non_finite, repair_spec, GmmControl::default()),
        Err(ClusteringError::NonFiniteInput { .. })
    ));
    assert!(matches!(
        fit_kmeans(
            &generated_clusters(),
            3,
            KMeansControl::new(1, 2, 0.0, 1, 1).unwrap()
        ),
        Err(ClusteringError::WorkLimit { .. })
    ));
    assert!(matches!(
        fit_gmm(
            &generated_clusters(),
            GmmSpec::new(
                3,
                CovarianceType::Full,
                1.0e-6,
                SingularComponentPolicy::default()
            )
            .unwrap(),
            GmmControl::new(1, 2, 0.0, 1).unwrap()
        ),
        Err(ClusteringError::WorkLimit { .. })
    ));
}
