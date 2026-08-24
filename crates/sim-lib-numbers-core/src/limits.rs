//! Canonical machine limits for fixed-width number domains.

use sim_kernel::Symbol;

use crate::domains;

/// Machine limits exposed by a scalar number domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MachineLimits {
    /// Distance from one to the next representable value, when bounded.
    pub epsilon: Option<f64>,
    /// Smallest finite value representable by the domain.
    pub minimum: Option<f64>,
    /// Largest finite value representable by the domain.
    pub maximum: Option<f64>,
    /// Smallest positive normal value for floating domains.
    pub minimum_positive: Option<f64>,
}

/// Returns canonical limits for a registered fixed-width domain.
///
/// Exact rationals are unbounded and have no machine epsilon, so all fields
/// are `None`. Unknown and extensible domains likewise return `None` rather
/// than borrowing another domain's constants.
pub fn machine_limits(domain: &Symbol) -> Option<MachineLimits> {
    if domain == &domains::f64() {
        Some(MachineLimits {
            epsilon: Some(f64::EPSILON),
            minimum: Some(f64::MIN),
            maximum: Some(f64::MAX),
            minimum_positive: Some(f64::MIN_POSITIVE),
        })
    } else if domain == &domains::f32() {
        Some(MachineLimits {
            epsilon: Some(f32::EPSILON as f64),
            minimum: Some(f32::MIN as f64),
            maximum: Some(f32::MAX as f64),
            minimum_positive: Some(f32::MIN_POSITIVE as f64),
        })
    } else if domain == &domains::i64() {
        Some(MachineLimits {
            epsilon: Some(1.0),
            minimum: Some(i64::MIN as f64),
            maximum: Some(i64::MAX as f64),
            minimum_positive: Some(1.0),
        })
    } else if domain == &domains::rational() {
        Some(MachineLimits {
            epsilon: None,
            minimum: None,
            maximum: None,
            minimum_positive: None,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_and_exact_domain_limits_are_explicit() {
        assert_eq!(
            machine_limits(&domains::f32()).unwrap().epsilon,
            Some(f32::EPSILON as f64)
        );
        assert_eq!(
            machine_limits(&domains::i64()).unwrap().minimum_positive,
            Some(1.0)
        );
        assert_eq!(machine_limits(&domains::rational()).unwrap().epsilon, None);
    }
}
