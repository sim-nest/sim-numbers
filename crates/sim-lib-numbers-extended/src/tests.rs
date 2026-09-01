use super::*;

fn s<T: RealScalar>(x: f64) -> T {
    T::from_f64(x).unwrap()
}
fn root<T: RealScalar>(mut lo: T, mut hi: T, f: impl Fn(T) -> T) -> T {
    for _ in 0..160 {
        let m = lo.add(hi).mul(s(0.5));
        if f(lo).to_f64().is_sign_negative() == f(m).to_f64().is_sign_negative() {
            lo = m
        } else {
            hi = m
        }
    }
    lo.add(hi).mul(s(0.5))
}
fn quad<T: RealScalar>(f: impl Fn(T) -> T, n: usize) -> T {
    let h = s::<T>(1.0 / n as f64);
    let mut z = T::ZERO;
    for i in 0..n {
        let x = s::<T>((i as f64 + 0.5) / n as f64);
        z = z.add(f(x).mul(h));
    }
    z
}
fn ode<T: RealScalar>(n: usize) -> T {
    let h = s::<T>(1.0 / n as f64);
    let mut y = T::ONE;
    for _ in 0..n {
        let k1 = y;
        let k2 = y.add(k1.mul(h).mul(s(0.5)));
        let k3 = y.add(k2.mul(h).mul(s(0.5)));
        let k4 = y.add(k3.mul(h));
        y = y.add(
            k1.add(k2.mul(s(2.0)))
                .add(k3.mul(s(2.0)))
                .add(k4)
                .mul(h.div(s(6.0))),
        );
    }
    y
}
fn horner<T: RealScalar>(x: T, c: &[T]) -> T {
    c.iter().rev().fold(T::ZERO, |y, &a| y.mul(x).add(a))
}
fn dominant_eigen<T: RealScalar>(a: [[T; 2]; 2]) -> T {
    let (mut x, mut y) = (T::ONE, T::ONE);
    for _ in 0..32 {
        let u = a[0][0].mul(x).add(a[0][1].mul(y));
        let v = a[1][0].mul(x).add(a[1][1].mul(y));
        let norm = u.mul(u).add(v.mul(v)).sqrt();
        x = u.div(norm);
        y = v.div(norm);
    }
    x.mul(a[0][0].mul(x).add(a[0][1].mul(y)))
        .add(y.mul(a[1][0].mul(x).add(a[1][1].mul(y))))
}

#[test]
fn transforms_normalize_and_preserve_cancellation() {
    let a = DoubleDouble::from_f64_exact(1e16) + DoubleDouble::ONE;
    let r = a - DoubleDouble::from_f64_exact(1e16);
    assert_eq!(r, DoubleDouble::ONE);
    for v in [a, r, DoubleDouble::PI] {
        let ulp = (v.hi.next_up() - v.hi).abs();
        assert!(v.lo.abs() <= 0.5 * ulp || v.lo == 0.0);
    }
}
#[test]
fn exact_bits_and_canonical_text_round_trip() {
    for v in [
        DoubleDouble::ZERO,
        DoubleDouble::PI,
        DoubleDouble::new(-0.0, 0.0),
        DoubleDouble::from_f64_exact(f64::INFINITY),
    ] {
        let text = v.canonical();
        assert_eq!(
            DoubleDouble::parse_canonical(&text).unwrap().to_bits(),
            v.to_bits()
        );
    }
}
#[test]
fn division_sqrt_and_elementary_accuracy() {
    let two = s::<DoubleDouble>(2.0);
    let root = two.sqrt();
    assert!((root * root - two).abs().to_f64().abs() < 1e-30);
    let x = DoubleDouble::new(0.7, 1e-18);
    assert!((x.exp().ln() - x).abs().to_f64() < 2e-30);
    let (sc, cc) = (x.sin(), x.cos());
    assert!((sc * sc + cc * cc - DoubleDouble::ONE).abs().to_f64() < 2e-30);
}
#[test]
fn finite_limits_and_non_finite_policy() {
    assert!(DoubleDouble::from_f64(f64::NAN).is_none());
    assert!(DoubleDouble::from_f64(f64::INFINITY).is_none());
    assert!(
        DoubleDouble::from_f64_exact(f64::INFINITY)
            .sqrt()
            .hi()
            .is_infinite()
    );
    assert!((DoubleDouble::from_f64_exact(-1.0).sqrt()).is_nan());
    assert_eq!(DoubleDouble::MAX.hi(), f64::MAX);
    assert_eq!(DoubleDouble::MIN_POSITIVE.hi(), f64::MIN_POSITIVE);
}
#[test]
fn exact_rational_order_and_wide_exponents() {
    let third = DoubleDouble::from_ratio_i128(1, 3).unwrap();
    assert!(third < DoubleDouble::from_f64_exact(0.334));
    assert!(third > DoubleDouble::from_f64_exact(0.333));
    assert_eq!(DoubleDouble::from_i128((1_i128 << 100) + 1).lo(), 1.0);
    let wide = DoubleDouble::from_f64_exact(1e200) + DoubleDouble::from_f64_exact(1e184);
    assert!(wide > DoubleDouble::from_f64_exact(1e200));
}
#[test]
fn promotion_edges_are_single_lattice_edges() {
    let rules = promotion_rules();
    assert_eq!(rules.len(), 2);
    assert!(rules.iter().any(|r| r.from_domain == domains::f64()));
    assert!(rules.iter().any(|r| r.from_domain == domains::rational()));
}
#[test]
fn shared_contracts_cover_five_kernel_families() {
    let rf = root::<f64>(1.0, 2.0, |x| x * x - 2.0);
    let rd = root::<DoubleDouble>(s(1.0), s(2.0), |x| x * x - s(2.0));
    assert!((rf - rd.to_f64()).abs() < 1e-15);
    let qf = quad::<f64>(|x| x * x, 4096);
    let qd = quad::<DoubleDouble>(|x| x * x, 4096);
    assert!((qf - qd.to_f64()).abs() < 1e-14);
    assert!((ode::<f64>(1024) - ode::<DoubleDouble>(1024).to_f64()).abs() < 1e-13);
    let cf = [s::<f64>(-1.0), s(3.0), s(-3.0), s(1.0)];
    let cd = [s::<DoubleDouble>(-1.0), s(3.0), s(-3.0), s(1.0)];
    assert!(horner(s::<DoubleDouble>(1.0 + 1e-12), &cd).abs() < s(1e-30));
    assert_eq!(horner(1.0 + 1e-12, &cf), 0.0);
    let af = [[s::<f64>(2.0), s(1.0)], [s(1.0), s(2.0)]];
    let ad = [[s::<DoubleDouble>(2.0), s(1.0)], [s(1.0), s(2.0)]];
    assert!((dominant_eigen(af) - dominant_eigen(ad).to_f64()).abs() < 1e-14);
}
#[test]
fn extended_corrects_known_f64_sign_loss() {
    // Keep the residual above the double-double precision floor while still
    // exercising a binary64 cancellation that loses the result's sign.
    let x: f64 = 1.0 + 7.2e-4;
    let f = (x - 1.0).powi(7);
    let expanded = x.powi(7) - 7.0 * x.powi(6) + 21.0 * x.powi(5) - 35.0 * x.powi(4)
        + 35.0 * x.powi(3)
        - 21.0 * x * x
        + 7.0 * x
        - 1.0;
    assert!(f > 0.0);
    assert!(expanded <= 0.0);
    let x = DoubleDouble::from_f64_exact(x);
    let e = x.powi(7) - s::<DoubleDouble>(7.0) * x.powi(6) + s::<DoubleDouble>(21.0) * x.powi(5)
        - s::<DoubleDouble>(35.0) * x.powi(4)
        + s::<DoubleDouble>(35.0) * x.powi(3)
        - s::<DoubleDouble>(21.0) * x * x
        + s::<DoubleDouble>(7.0) * x
        - DoubleDouble::ONE;
    assert!(e > DoubleDouble::ZERO);
}
// conformance: extended-number tests prove normalized arithmetic and convergence behavior.
