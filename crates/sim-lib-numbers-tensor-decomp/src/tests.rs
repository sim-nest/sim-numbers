use super::*;
fn close(x: f64, t: f64) {
    assert!(x < t, "{x} >= {t}")
}
#[test]
fn qr_diagonal_pivoted_hilbert_scaled_and_immutable() {
    for a in [
        vec![3., 0., 0., 2.],
        vec![1., 0.5, 1. / 3., 0.5, 1. / 3., 0.25, 1. / 3., 0.25, 0.2],
        vec![1e8, 2., 3., 4e-8],
    ] {
        let n = (a.len() as f64).sqrt() as usize;
        let before = a.clone();
        let out = qr_f64(
            &a,
            n,
            n,
            QrPlan {
                column_pivoting: true,
                reconstruction_tolerance: 1e-6,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a, before);
        close(out.evidence.orthogonality_residual, 1e-8);
        close(out.evidence.reconstruction_residual, 1e-5);
        assert_eq!(out.permutation.len(), n)
    }
}
#[test]
fn qr_permutation_and_work_limit() {
    let a = [0., 2., 1., 3.];
    let q = qr_f64(
        &a,
        2,
        2,
        QrPlan {
            column_pivoting: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(q.permutation, vec![1, 0]);
    assert!(matches!(
        qr_f64(
            &a,
            2,
            2,
            QrPlan {
                max_work: 1,
                ..Default::default()
            }
        ),
        Err(DecompositionError::WorkLimit)
    ));
}
#[test]
fn eigen_diagonal_rotation_repeated_clustered_and_scaled() {
    let c = 0.6;
    let s = 0.8;
    for a in [
        vec![3., 0., 0., 1.],
        vec![
            3. * c * c + s * s,
            (3. - 1.) * c * s,
            (3. - 1.) * c * s,
            3. * s * s + c * c,
        ],
        vec![2., 0., 0., 2.],
        vec![1., 1e-10, 1e-10, 1. + 1e-12],
        vec![1e8, 0., 0., -1e8],
    ] {
        let before = a.clone();
        let e = symmetric_eigen_f64(
            &a,
            2,
            EigenPlan {
                symmetry_tolerance: Some(1e-14),
                reconstruction_tolerance: 1e-6,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a, before);
        assert!(e.eigenvalues[0] >= e.eigenvalues[1]);
        close(e.evidence.orthogonality_residual, 1e-8);
        close(
            e.evidence.reconstruction_residual,
            1e-5 * matrix_norm(&a).max(1.),
        );
    }
}
#[test]
fn eigen_refuses_asymmetry_and_honors_iteration_limit() {
    let a = [1., 0.1, 0.2, 2.];
    assert!(matches!(
        symmetric_eigen_f64(&a, 2, EigenPlan::default()),
        Err(DecompositionError::Asymmetric { .. })
    ));
    let a = [1., 1., 1., 2.];
    assert!(matches!(
        symmetric_eigen_f64(
            &a,
            2,
            EigenPlan {
                max_iterations: 0,
                symmetry_tolerance: Some(0.),
                ..Default::default()
            }
        ),
        Err(DecompositionError::InvalidPlan(_))
    ));
}

#[test]
fn eigen_householder_stage_handles_hilbert_and_permuted_three_by_three() {
    for a in [
        vec![
            1.0,
            0.5,
            1.0 / 3.0,
            0.5,
            1.0 / 3.0,
            0.25,
            1.0 / 3.0,
            0.25,
            0.2,
        ],
        vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 2.0],
    ] {
        let e = symmetric_eigen_f64(
            &a,
            3,
            EigenPlan {
                symmetry_tolerance: Some(0.0),
                max_iterations: 256,
                reconstruction_tolerance: 1e-9,
                ..Default::default()
            },
        )
        .unwrap();
        close(e.evidence.reconstruction_residual, 1e-8);
        close(e.evidence.orthogonality_residual, 1e-8);
    }
}

#[test]
fn schur_covers_normal_non_normal_defective_and_complex_pairs() {
    for (a, n) in [
        (vec![0.0, -1.0, 1.0, 0.0], 2),
        (vec![1.0, 100.0, 0.0, 2.0], 2),
        (vec![2.0, 1.0, 0.0, 2.0], 2),
        (vec![1.0, 1.0, 0.0, 1.0e-10, 1.0, 1.0, 0.0, 1.0e-10, 1.0], 3),
    ] {
        let out = real_schur_f64(&a, n, SchurPlan::default()).unwrap();
        assert_eq!(out.status, FactorStatus::Complete);
        assert_eq!(out.eigenvalues.len(), n);
        close(out.evidence.reconstruction_residual, 1.0e-7);
        close(out.evidence.orthogonality_residual, 1.0e-7);
    }
}

#[test]
fn svd_covers_rectangular_rank_deficient_repeated_scaled_and_permuted() {
    for (a, rows, cols) in [
        (vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3, 2),
        (vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2),
        (vec![0.0, 1.0e12, 1.0e-12, 0.0], 2, 2),
        (vec![0.0, 1.0, 1.0, 0.0], 2, 2),
        (vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0], 2, 3),
    ] {
        let out = svd_f64(
            &a,
            rows,
            cols,
            SvdPlan {
                vectors: VectorForm::Full,
                reconstruction_tolerance: 1.0e-8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.status, FactorStatus::Complete);
        assert!(out.singular_values.windows(2).all(|x| x[0] >= x[1]));
        close(out.evidence.reconstruction_residual, 1.0e-8);
    }
}

#[test]
fn svd_values_only_still_carries_a_reconstruction_certificate() {
    for (a, rows, cols) in [
        (vec![3.0, 0.0, 0.0, 1.0], 2, 2),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
    ] {
        let out = svd_f64(
            &a,
            rows,
            cols,
            SvdPlan {
                vectors: VectorForm::None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(out.u.is_none() && out.v.is_none());
        close(out.evidence.reconstruction_residual, 1.0e-9);
    }
}

#[test]
fn svd_derived_solves_share_one_cutoff_and_retain_near_rank() {
    // Squaring this matrix into normal equations loses the small direction much
    // earlier; direct column rotations retain it under the declared cutoff.
    let a = [1.0, 1.0, 1.0, 1.0 + 1.0e-10];
    let s = svd_f64(
        &a,
        2,
        2,
        SvdPlan {
            vectors: VectorForm::Full,
            ..Default::default()
        },
    )
    .unwrap();
    let cutoff = SingularCutoff::new(1.0e-12).unwrap();
    assert_eq!(numerical_rank(&s, cutoff).unwrap(), 2);
    assert!(condition_2(&s, cutoff).unwrap().is_finite());
    let pinv = pseudoinverse(&s, cutoff).unwrap();
    assert_eq!(pinv.len(), 4);
    let x = least_squares(&s, &[2.0, 2.0 + 1.0e-10], cutoff).unwrap();
    close((x[0] - 1.0).abs(), 1.0e-4);
    close((x[1] - 1.0).abs(), 1.0e-4);
    assert!(null_space(&s, cutoff).unwrap().is_empty());

    let deficient = svd_f64(
        &[1.0, 2.0, 2.0, 4.0],
        2,
        2,
        SvdPlan {
            vectors: VectorForm::Full,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(numerical_rank(&deficient, cutoff).unwrap(), 1);
    assert!(condition_2(&deficient, cutoff).unwrap().is_infinite());
    assert_eq!(null_space(&deficient, cutoff).unwrap().len(), 2);
}

#[test]
fn exhaustion_is_labelled_partial_and_never_admitted_to_solves() {
    let a = [1.0, 2.0, 3.0, 4.0, 7.0, 11.0, 5.0, 13.0, 17.0];
    let partial = svd_f64(
        &a,
        3,
        3,
        SvdPlan {
            max_iterations: 1,
            return_partial: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(partial.status, FactorStatus::Partial);
    assert!(matches!(
        numerical_rank(&partial, SingularCutoff(1.0e-12)),
        Err(DecompositionError::NoConvergence)
    ));
    assert!(matches!(
        svd_f64(
            &a,
            3,
            3,
            SvdPlan {
                max_iterations: 1,
                ..Default::default()
            }
        ),
        Err(DecompositionError::NoConvergence)
    ));
}

#[test]
fn provider_descriptors_and_admission_check_identity_shape_order_and_residual() {
    let a = [3.0, 0.0, 0.0, 1.0];
    let s = svd_f64(&a, 2, 2, SvdPlan::default()).unwrap();
    let execution = ExecutionIdentity::new("test", "provider", "invocation-1").unwrap();
    let admitted = admit_provider_svd(&a, 2, 2, &s, execution.clone(), 1.0e-10).unwrap();
    assert_eq!(admitted.execution, execution);
    assert_eq!(svd_operation(2, 2, VectorForm::Thin).output_shapes.len(), 3);
    assert_eq!(schur_operation(2).output_shapes.len(), 3);
    let mut invalid = s.clone();
    invalid.singular_values.swap(0, 1);
    assert!(admit_provider_svd(&a, 2, 2, &invalid, execution, 1.0e-10).is_err());
}
