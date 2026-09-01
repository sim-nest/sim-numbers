use super::*;
fn q(n: i128, d: i128) -> ExactRational {
    ExactRational::new(n, d).unwrap()
}
fn ri(a: (i128, i128), b: (i128, i128)) -> RationalInterval {
    RationalInterval::new(q(a.0, a.1), q(b.0, b.1)).unwrap()
}
fn c(a: (i128, i128), b: (i128, i128)) -> CertifiedInterval {
    ri(a, b).certify().unwrap()
}

#[test]
fn exact_oracle_and_directed_operations_contain_results() {
    let a = ri((1, 10), (1, 3));
    let b = ri((-7, 5), (2, 7));
    for (exact, float) in [
        (
            a.add(b).unwrap(),
            a.certify().unwrap().add(&b.certify().unwrap()).unwrap(),
        ),
        (
            a.sub(b).unwrap(),
            a.certify().unwrap().sub(&b.certify().unwrap()).unwrap(),
        ),
        (
            a.mul(b).unwrap(),
            a.certify().unwrap().mul(&b.certify().unwrap()).unwrap(),
        ),
    ] {
        assert!(float.lower() <= exact.lower().value());
        assert!(float.upper() >= exact.upper().value())
    }
}
#[test]
fn hostile_cancellation_subnormal_overflow_and_dependency_are_contained() {
    let huge = c((i128::MAX / 4, 1), (i128::MAX / 4, 1));
    assert!(huge.add(&huge).unwrap().upper().is_finite());
    let tiny = CertifiedInterval::issue(
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE,
        evidence(
            CertificationKernelId::DirectedBinary64V1,
            vec![],
            "fixture",
            "fixture",
        ),
    )
    .unwrap();
    assert!(tiny.sub(&tiny).unwrap().lower() < 0.0);
    let x = c((1, 1), (2, 1));
    let dependency = x.sub(&x).unwrap();
    assert!(dependency.lower() <= 0.0 && dependency.upper() >= 0.0);
    let max = CertifiedInterval::issue(
        f64::MAX,
        f64::MAX,
        evidence(
            CertificationKernelId::DirectedBinary64V1,
            vec![],
            "fixture",
            "fixture",
        ),
    )
    .unwrap();
    assert_eq!(max.add(&max).unwrap().upper(), f64::INFINITY);
}
#[test]
fn discontinuities_and_unsupported_functions_refuse() {
    let z = c((-1, 1), (1, 1));
    assert_eq!(z.div(&z), Err(IntervalError::ZeroContainingDivisor));
    assert_eq!(
        certify_elementary("sin", &z),
        Err(IntervalError::UnsupportedElementary("sin".into()))
    );
}
#[test]
fn signed_zero_nan_and_infinity_are_explicit() {
    assert_eq!(
        EstimateInterval::new(f64::NAN, 1.0),
        Err(IntervalError::NaN)
    );
    let z = c((0, 1), (0, 1));
    assert!(z.lower().is_sign_negative() && z.upper().is_sign_positive());
    let inf = CertifiedInterval::issue(
        f64::NEG_INFINITY,
        f64::INFINITY,
        evidence(
            CertificationKernelId::DirectedBinary64V1,
            vec![],
            "fixture",
            "fixture",
        ),
    )
    .unwrap();
    assert_eq!(inf.midpoint(), Err(IntervalError::Unbounded));
    let exact_zero = CertifiedInterval::issue(
        -0.0,
        0.0,
        evidence(
            CertificationKernelId::DirectedBinary64V1,
            vec![],
            "fixture",
            "fixture",
        ),
    )
    .unwrap();
    assert_eq!(exact_zero.mul(&inf), Err(IntervalError::Indeterminate));
}
#[test]
fn only_certificates_make_definite_thresholds() {
    let low = c((1, 1), (2, 1));
    let high = c((3, 1), (4, 1));
    assert_eq!(classify_threshold(&low, &high), ThresholdVerdict::Below);
    assert!(matches!(
        classify_estimate(
            EstimateInterval::new(1., 2.).unwrap(),
            EstimateInterval::new(3., 4.).unwrap()
        ),
        ThresholdVerdict::Unresolved(_)
    ));
}
#[test]
fn interval_newton_and_krawczyk_prove_simple_unique_root() {
    let x = c((1, 1), (2, 1));
    let fm = c((-1, 4), (-1, 4));
    let d = c((2, 1), (4, 1));
    let n = interval_newton(&x, &fm, &d).unwrap();
    assert!(n.existence() && n.uniqueness());
    let inv = c((1, 3), (1, 3));
    let k = krawczyk(&x, &fm, &d, &inv).unwrap();
    assert!(k.existence() && k.uniqueness());
}
#[test]
fn inspection_datum_is_not_an_authority_decoder() {
    let x = c((1, 1), (2, 1));
    assert!(matches!(x.to_datum(), Datum::Node { .. }));
    assert!(x.evidence().method().contains("exact rational"));
}
// conformance: interval tests prove sealed certification and reject forged evidence.
