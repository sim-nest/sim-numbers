//! Standardized moments and distribution-distance statistics.

use super::{StatsError, StatsResult, mean, validate_values};

/// Explicit finite-sample estimator convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MomentConvention {
    /// Population central moments divided by population variance powers.
    Population,
    /// Fisher-Pearson adjusted skewness and excess kurtosis.
    UnbiasedSample,
}
/// Standardized third and fourth moment report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandardizedMoments {
    /// Third standardized moment.
    pub skewness: f64,
    /// Fourth standardized moment (not excess for population convention).
    pub kurtosis: f64,
    /// Applied estimator convention.
    pub convention: MomentConvention,
    /// Sample count.
    pub samples: usize,
}
/// Computes standardized third and fourth moments under an explicit convention.
pub fn standardized_moments(
    values: &[f64],
    convention: MomentConvention,
) -> StatsResult<StandardizedMoments> {
    validate_values("standardized_moments", values)?;
    let minimum = if convention == MomentConvention::UnbiasedSample {
        4
    } else {
        2
    };
    if values.len() < minimum {
        return Err(StatsError::InsufficientInput {
            metric: "standardized_moments",
            minimum,
            actual: values.len(),
        });
    }
    let center = mean(values)?;
    let n = values.len() as f64;
    let m2 = values.iter().map(|x| (x - center).powi(2)).sum::<f64>() / n;
    if m2 == 0.0 {
        return Err(StatsError::InvalidControl {
            field: "variance",
            reason: "standardized moments require positive variance",
        });
    }
    let m3 = values.iter().map(|x| (x - center).powi(3)).sum::<f64>() / n;
    let m4 = values.iter().map(|x| (x - center).powi(4)).sum::<f64>() / n;
    let (skewness, kurtosis) = match convention {
        MomentConvention::Population => (m3 / m2.powf(1.5), m4 / (m2 * m2)),
        MomentConvention::UnbiasedSample => {
            let g1 = m3 / m2.powf(1.5);
            let g2 = m4 / (m2 * m2) - 3.0;
            (
                (n * (n - 1.0)).sqrt() / (n - 2.0) * g1,
                (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0),
            )
        }
    };
    Ok(StandardizedMoments {
        skewness,
        kurtosis,
        convention,
        samples: values.len(),
    })
}

/// Kolmogorov-Smirnov evaluation policy and applicability identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KsMethod {
    /// Exact empirical statistic; no approximate probability is asserted.
    ExactStatistic,
    /// Standard asymptotic Kolmogorov tail approximation.
    Asymptotic,
}
/// One- or two-sample Kolmogorov-Smirnov report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KsResult {
    /// Supremum empirical-CDF distance.
    pub statistic: f64,
    /// Optional asymptotic tail probability.
    pub p_value: Option<f64>,
    /// Applied method.
    pub method: KsMethod,
    /// Effective sample size used by the approximation.
    pub effective_samples: f64,
}

/// Computes a one-sample KS statistic against a caller-supplied continuous CDF.
pub fn kolmogorov_smirnov_one_sample(
    values: &[f64],
    cdf: impl Fn(f64) -> f64,
    method: KsMethod,
) -> StatsResult<KsResult> {
    validate_values("kolmogorov_smirnov_one_sample", values)?;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let mut d = 0.0_f64;
    for (i, x) in sorted.into_iter().enumerate() {
        let p = cdf(x);
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange {
                metric: "kolmogorov_smirnov_one_sample",
                index: Some(i),
                value: p,
            });
        }
        d = d
            .max(((i + 1) as f64 / n - p).abs())
            .max((p - i as f64 / n).abs());
    }
    Ok(ks_result(d, n, method))
}
/// Computes the two-sample KS statistic. Inputs must represent independent samples.
pub fn kolmogorov_smirnov_two_sample(
    left: &[f64],
    right: &[f64],
    method: KsMethod,
) -> StatsResult<KsResult> {
    validate_values("kolmogorov_smirnov_two_sample:left", left)?;
    validate_values("kolmogorov_smirnov_two_sample:right", right)?;
    let mut a = left.to_vec();
    let mut b = right.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (mut i, mut j, mut d) = (0, 0, 0.0_f64);
    while i < a.len() || j < b.len() {
        let x = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => {
                if x.total_cmp(y).is_le() {
                    *x
                } else {
                    *y
                }
            }
            (Some(x), None) => *x,
            (None, Some(y)) => *y,
            (None, None) => break,
        };
        while i < a.len() && a[i] <= x {
            i += 1
        }
        while j < b.len() && b[j] <= x {
            j += 1
        }
        d = d.max((i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs());
    }
    let effective = (a.len() as f64 * b.len() as f64) / (a.len() + b.len()) as f64;
    Ok(ks_result(d, effective, method))
}
fn ks_result(d: f64, effective: f64, method: KsMethod) -> KsResult {
    let p_value = (method == KsMethod::Asymptotic).then(|| {
        let lambda = (effective.sqrt() + 0.12 + 0.11 / effective.sqrt()) * d;
        (1..=100)
            .map(|k| {
                let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
                2.0 * sign * (-2.0 * (k * k) as f64 * lambda * lambda).exp()
            })
            .sum::<f64>()
            .clamp(0.0, 1.0)
    });
    KsResult {
        statistic: d,
        p_value,
        method,
        effective_samples: effective,
    }
}
