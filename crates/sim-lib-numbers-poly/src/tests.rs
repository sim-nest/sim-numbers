use super::*;

fn r(n: i128, d: i128) -> Rational {
    Rational::new(n, d).unwrap()
}
#[test]
fn canonical_exponent_domains_are_distinct() {
    assert_eq!(Polynomial::new(vec![]).unwrap().coefficients(), &[0.]);
    let l = LaurentPolynomial::new(-3, vec![0., 2., 0.]).unwrap();
    assert_eq!(l.minimum_exponent, -2);
    assert_eq!(l.coefficients(), &[2.]);
    let p = PuiseuxPolynomial::new(
        RationalExponent::new(-1, 2).unwrap(),
        RationalExponent::new(1, 3).unwrap(),
        vec![0., 4.],
    )
    .unwrap();
    assert_eq!(p.minimum_exponent, RationalExponent::new(-1, 6).unwrap());
    assert_eq!(p.coefficients(), &[4.])
}
#[test]
fn exact_identities_division_and_gcd() {
    let a = ExactPolynomial::new(vec![r(-1, 1), r(0, 1), r(1, 1)]);
    let b = ExactPolynomial::new(vec![r(-1, 1), r(1, 1)]);
    let (q, rem) = a.div_rem(&b).unwrap();
    assert_eq!(q.coefficients(), &[r(1, 1), r(1, 1)]);
    assert_eq!(rem.coefficients(), &[Rational::ZERO]);
    let g = ExactPolynomial::gcd(a, b.clone()).unwrap();
    assert_eq!(g, b)
}
#[test]
fn horner_derivative_and_integral_carry_evidence() {
    let p = Polynomial::new(vec![1., -3., 2.]).unwrap();
    let e = p.evaluate(2.);
    let c = p.evaluate_compensated(2.);
    assert_eq!(e.value, 3.);
    assert!(e.absolute_error_bound >= 0.);
    assert!((c.value - 3.).abs() <= c.absolute_error_bound);
    assert_eq!(p.derivative().coefficients(), &[-3., 4.]);
    assert_eq!(p.antiderivative(7.).unwrap().derivative(), p);
    assert_eq!(p.add(&p).coefficients(), &[2., -6., 4.]);
    assert_eq!(
        p.mul(&Polynomial::new(vec![1., 1.]).unwrap())
            .coefficients(),
        &[1., -2., -1., 2.]
    )
}
#[test]
fn roots_cover_repetition_clusters_sensitivity_and_imaginary_pairs() {
    let repeated = Polynomial::from_roots(&[1., 1., 1.])
        .unwrap()
        .roots(RootPlan {
            cluster_tolerance: 1e-3,
            ..RootPlan::default()
        })
        .unwrap();
    assert_eq!(repeated.roots.len(), 3);
    assert!(repeated.roots.iter().all(|x| x.multiplicity >= 2));
    let imag = Polynomial::new(vec![1., 0., 1.])
        .unwrap()
        .roots(RootPlan::default())
        .unwrap();
    assert!(imag.roots.iter().any(|x| x.root.im().abs() > 0.9));
    assert!(imag.roots.iter().all(|x| x.backward_error < 1e-7));
    let wilkinson = Polynomial::from_roots(&(1..=8).map(|x| x as f64).collect::<Vec<_>>()).unwrap();
    let roots = wilkinson
        .roots(RootPlan {
            cluster_tolerance: 1e-5,
            ..RootPlan::default()
        })
        .unwrap();
    assert_eq!(roots.roots.len(), 8);
    assert!(roots.roots.iter().all(|x| x.backward_error < 1e-4));
    let clustered = Polynomial::from_roots(&[1., 1. + 1e-8, 2.])
        .unwrap()
        .roots(RootPlan {
            cluster_tolerance: 1e-4,
            ..RootPlan::default()
        })
        .unwrap();
    assert!(clustered.roots.iter().any(|x| x.multiplicity >= 2))
}
#[test]
fn pade_matches_exponential_series_and_rejects_singular_system() {
    let mut series = vec![1.];
    for k in 1..=6 {
        series.push(series[k - 1] / k as f64)
    }
    let p = pade(&series, 3, 3, 1e-12).unwrap();
    assert_eq!(p.rank, 3);
    for &x in &[-0.5, 0., 0.5] {
        let value = p.numerator.evaluate(x).value / p.denominator.evaluate(x).value;
        assert!((value - x.exp()).abs() < 1e-6)
    }
    let singular = vec![1.; 5];
    assert!(matches!(
        pade(&singular, 2, 2, 1e-12),
        Err(PolynomialError::RankDeficient { .. })
    ))
}
