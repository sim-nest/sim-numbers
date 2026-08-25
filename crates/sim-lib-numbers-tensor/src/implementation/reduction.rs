//! Shared, auditable floating-point reduction policies.
/// Explicit addition policy for Tensor `sum` and `cumsum`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SumMode {
    /// Sequential.
    Naive,
    /// Balanced tree.
    Pairwise,
    /// Neumaier compensated.
    Neumaier,
}
/// Reduces a slice with the named policy.
pub fn sum_f64(v: &[f64], m: SumMode) -> f64 {
    match m {
        SumMode::Naive => v.iter().sum(),
        SumMode::Pairwise => {
            if v.len() < 2 {
                v.first().copied().unwrap_or(0.)
            } else {
                let n = v.len() / 2;
                sum_f64(&v[..n], m) + sum_f64(&v[n..], m)
            }
        }
        SumMode::Neumaier => {
            let (mut s, mut c) = (0., 0.);
            for &x in v {
                let t = s + x;
                c += if s.abs() >= x.abs() {
                    (s - t) + x
                } else {
                    (x - t) + s
                };
                s = t
            }
            s + c
        }
    }
}
/// Produces prefix sums with the named policy.
pub fn cumsum_f64(v: &[f64], m: SumMode) -> Vec<f64> {
    match m {
        SumMode::Naive => {
            let mut s = 0.;
            v.iter()
                .map(|&x| {
                    s += x;
                    s
                })
                .collect()
        }
        SumMode::Pairwise => (1..=v.len()).map(|n| sum_f64(&v[..n], m)).collect(),
        SumMode::Neumaier => {
            let (mut s, mut c) = (0., 0.);
            v.iter()
                .map(|&x| {
                    let t = s + x;
                    c += if s.abs() >= x.abs() {
                        (s - t) + x
                    } else {
                        (x - t) + s
                    };
                    s = t;
                    s + c
                })
                .collect()
        }
    }
}
