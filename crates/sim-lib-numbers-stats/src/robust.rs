//! Robust dispersion and deterministic uncertainty intervals for comparisons.

use super::{StatsError, StatsResult, mean, validate_values};
use crate::exact_quantile;

/// Controls a deterministic bootstrap of the candidate-minus-baseline mean.
///
/// One work unit is one sampled observation. `max_work` therefore bounds the
/// exact resampling cost before allocation or sampling begins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapControl {
    /// Seed for the platform-independent SplitMix64 resampler.
    pub seed: u64,
    /// Number of bootstrap replicates to retain.
    pub resamples: usize,
    /// Central interval mass, strictly between zero and one.
    pub confidence_level: f64,
    /// Maximum admitted sampled observations across all replicates.
    pub max_work: u64,
}

impl BootstrapControl {
    /// Builds and validates a bootstrap control.
    pub fn new(
        seed: u64,
        resamples: usize,
        confidence_level: f64,
        max_work: u64,
    ) -> StatsResult<Self> {
        let control = Self {
            seed,
            resamples,
            confidence_level,
            max_work,
        };
        control.validate()?;
        Ok(control)
    }

    fn validate(self) -> StatsResult<()> {
        if self.resamples < 2 {
            return Err(StatsError::InvalidControl {
                field: "resamples",
                reason: "must be at least two",
            });
        }
        if !self.confidence_level.is_finite() || !(0.0..1.0).contains(&self.confidence_level) {
            return Err(StatsError::InvalidControl {
                field: "confidence_level",
                reason: "must be finite and strictly between zero and one",
            });
        }
        Ok(())
    }
}

/// A percentile bootstrap interval for the candidate-minus-baseline mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapEffectInterval {
    /// Mean candidate effect minus mean baseline effect in source units.
    pub point_effect: f64,
    /// Lower endpoint of the central percentile interval.
    pub lower: f64,
    /// Upper endpoint of the central percentile interval.
    pub upper: f64,
    /// Central interval mass requested by the caller.
    pub confidence_level: f64,
    /// Seed used by the resampler.
    pub seed: u64,
    /// Number of retained bootstrap replicates.
    pub resamples: usize,
    /// Number of source baseline observations.
    pub baseline_samples: usize,
    /// Number of source candidate observations.
    pub candidate_samples: usize,
}

/// Computes the raw median absolute deviation from the sample median.
///
/// This returns MAD in source units without a normal-distribution scaling
/// factor, leaving benchmark comparison policy explicit about its threshold.
pub fn median_absolute_deviation(values: &[f64]) -> StatsResult<f64> {
    validate_values("median_absolute_deviation", values)?;
    let median = exact_quantile(values, 0.5).map_err(|_| StatsError::InvalidControl {
        field: "quantile",
        reason: "internal median quantile must remain valid",
    })?;
    let deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    exact_quantile(&deviations, 0.5).map_err(|_| StatsError::InvalidControl {
        field: "quantile",
        reason: "internal deviation quantile must remain valid",
    })
}

/// Bootstraps the difference between candidate and baseline arithmetic means.
///
/// The two samples are resampled independently with replacement. This matches
/// the benchmark comparison policy's interleaved but not necessarily paired
/// observations. The same inputs and control produce bit-identical results on
/// every target with IEEE-754 `f64` arithmetic.
pub fn bootstrap_mean_difference_interval(
    baseline: &[f64],
    candidate: &[f64],
    control: BootstrapControl,
) -> StatsResult<BootstrapEffectInterval> {
    validate_values("bootstrap baseline", baseline)?;
    validate_values("bootstrap candidate", candidate)?;
    control.validate()?;

    let observations = baseline
        .len()
        .checked_add(candidate.len())
        .and_then(|count| u64::try_from(count).ok())
        .ok_or(StatsError::WorkLimitExceeded {
            required: u64::MAX,
            limit: control.max_work,
        })?;
    let required = observations.checked_mul(control.resamples as u64).ok_or(
        StatsError::WorkLimitExceeded {
            required: u64::MAX,
            limit: control.max_work,
        },
    )?;
    if required > control.max_work {
        return Err(StatsError::WorkLimitExceeded {
            required,
            limit: control.max_work,
        });
    }

    let mut rng = SplitMix64(control.seed);
    let mut effects = Vec::with_capacity(control.resamples);
    for _ in 0..control.resamples {
        let baseline_mean = resampled_mean(baseline, &mut rng);
        let candidate_mean = resampled_mean(candidate, &mut rng);
        effects.push(candidate_mean - baseline_mean);
    }
    let tail = (1.0 - control.confidence_level) / 2.0;
    let lower = exact_quantile(&effects, tail).expect("validated non-empty bootstrap quantile");
    let upper =
        exact_quantile(&effects, 1.0 - tail).expect("validated non-empty bootstrap quantile");

    Ok(BootstrapEffectInterval {
        point_effect: mean(candidate)? - mean(baseline)?,
        lower,
        upper,
        confidence_level: control.confidence_level,
        seed: control.seed,
        resamples: control.resamples,
        baseline_samples: baseline.len(),
        candidate_samples: candidate.len(),
    })
}

fn resampled_mean(values: &[f64], rng: &mut SplitMix64) -> f64 {
    let sum = (0..values.len())
        .map(|_| values[rng.index(values.len())])
        .sum::<f64>();
    sum / values.len() as f64
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        ((u128::from(self.next()) * len as u128) >> 64) as usize
    }
}
