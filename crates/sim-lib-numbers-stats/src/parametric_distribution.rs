//! Normal and Student-t distributions composed from shared numerical owners.

use super::{StatsError, StatsResult};
use sim_lib_numbers_special::{erfc, log_gamma, regularized_beta};

fn valid_probability(metric: &'static str, value: f64) -> StatsResult<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(StatsError::ProbabilityOutOfRange {
            metric,
            index: None,
            value,
        })
    }
}
fn quantile_root(mut lower: f64, mut upper: f64, p: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    for _ in 0..192 {
        let middle = lower + (upper - lower) / 2.0;
        if cdf(middle) < p {
            lower = middle;
        } else {
            upper = middle;
        }
        if upper - lower <= 2.0 * f64::EPSILON * middle.abs().max(1.0) {
            break;
        }
    }
    lower + (upper - lower) / 2.0
}

/// Standard normal probability density.
pub fn normal_density(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}
/// Standard normal cumulative probability.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2).value
}
/// Standard normal survival probability, evaluated directly in the positive tail.
pub fn normal_survival(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2).value
}
/// Standard normal quantile, found by bounded monotone inversion.
pub fn normal_quantile(p: f64) -> StatsResult<f64> {
    valid_probability("normal_quantile", p)?;
    if p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    Ok(quantile_root(-40.0, 40.0, p, normal_cdf))
}

/// Student-t probability density for positive degrees of freedom.
pub fn student_t_density(x: f64, v: f64) -> StatsResult<f64> {
    if !(v > 0.0 && v.is_finite()) {
        return Err(StatsError::InvalidControl {
            field: "degrees_of_freedom",
            reason: "must be finite and positive",
        });
    }
    let lg1 = log_gamma((v + 1.0) / 2.0)
        .map_err(|_| StatsError::InvalidControl {
            field: "degrees_of_freedom",
            reason: "gamma evaluation failed",
        })?
        .value;
    let lg0 = log_gamma(v / 2.0)
        .map_err(|_| StatsError::InvalidControl {
            field: "degrees_of_freedom",
            reason: "gamma evaluation failed",
        })?
        .value;
    Ok(
        (lg1 - lg0 - 0.5 * (v * std::f64::consts::PI).ln() - 0.5 * (v + 1.0) * (x * x / v).ln_1p())
            .exp(),
    )
}
fn student_tail(x: f64, v: f64) -> StatsResult<f64> {
    if !(v > 0.0 && v.is_finite()) {
        return Err(StatsError::InvalidControl {
            field: "degrees_of_freedom",
            reason: "must be finite and positive",
        });
    }
    regularized_beta(v / 2.0, 0.5, v / (v + x * x))
        .map(|r| 0.5 * r.value)
        .map_err(|_| StatsError::InvalidControl {
            field: "degrees_of_freedom",
            reason: "beta evaluation failed",
        })
}
/// Student-t cumulative probability.
pub fn student_t_cdf(x: f64, v: f64) -> StatsResult<f64> {
    let tail = student_tail(x, v)?;
    Ok(if x >= 0.0 { 1.0 - tail } else { tail })
}
/// Student-t survival probability, using the direct beta tail for positive values.
pub fn student_t_survival(x: f64, v: f64) -> StatsResult<f64> {
    let tail = student_tail(x, v)?;
    Ok(if x >= 0.0 { tail } else { 1.0 - tail })
}
/// Student-t quantile, found by bounded monotone inversion.
pub fn student_t_quantile(p: f64, v: f64) -> StatsResult<f64> {
    valid_probability("student_t_quantile", p)?;
    if p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    student_t_density(0.0, v)?;
    let mut bound = 1.0;
    while (student_t_cdf(-bound, v)? > p || student_t_cdf(bound, v)? < p) && bound < 1e16 {
        bound *= 2.0;
    }
    Ok(quantile_root(-bound, bound, p, |x| {
        student_t_cdf(x, v).unwrap_or(f64::NAN)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64, t: f64) {
        assert!((a - b).abs() < t * b.abs().max(1.0), "{a} != {b}");
    }
    #[test]
    fn normal_center_tails_and_inversion() {
        close(normal_density(0.0), 0.3989422804014327, 1e-15);
        close(normal_cdf(0.0), 0.5, 1e-15);
        assert!(normal_survival(10.0) > 0.0);
        for p in [1e-10, 0.01, 0.5, 0.99, 1.0 - 1e-10] {
            close(normal_cdf(normal_quantile(p).unwrap()), p, 3e-8);
        }
    }
    #[test]
    fn student_symmetry_tails_and_inversion() {
        for v in [1.0, 5.0, 30.0] {
            close(student_t_cdf(0.0, v).unwrap(), 0.5, 1e-14);
            assert!(student_t_survival(20.0, v).unwrap() > 0.0);
            for p in [1e-5, 0.1, 0.9, 1.0 - 1e-5] {
                let x = student_t_quantile(p, v).unwrap();
                close(student_t_cdf(x, v).unwrap(), p, 2e-11);
            }
        }
    }
}
