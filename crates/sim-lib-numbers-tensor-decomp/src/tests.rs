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
