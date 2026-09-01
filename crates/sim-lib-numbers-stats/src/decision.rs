//! Bounded, deterministic inference primitives for sequential study decisions.

use super::{BootstrapControl, BootstrapEffectInterval, StatsError, StatsResult, exact_quantile};
use crate::SeededSampler;

/// A two-sided Clopper--Pearson interval for a finite binary count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryInterval {
    /// Observed successes.
    pub successes: u64,
    /// Observed trials.
    pub trials: u64,
    /// Declared central confidence mass.
    pub confidence_level: f64,
    /// Exact lower endpoint.
    pub lower: f64,
    /// Exact upper endpoint.
    pub upper: f64,
}

/// Computes an exact equal-tailed finite-count binary interval.
///
/// The endpoints invert binomial tail probabilities and use no normal or
/// other large-sample approximation.
pub fn exact_binary_interval(
    successes: u64,
    trials: u64,
    confidence_level: f64,
) -> StatsResult<BinaryInterval> {
    confidence(confidence_level)?;
    if trials == 0 {
        return Err(StatsError::ZeroTotal {
            label: "binary trials",
        });
    }
    if successes > trials {
        return Err(StatsError::InvalidControl {
            field: "successes",
            reason: "must not exceed trials",
        });
    }
    let alpha = (1.0 - confidence_level) / 2.0;
    let lower = if successes == 0 {
        0.0
    } else {
        bisect_probability(|p| binomial_upper_tail(successes, trials, p), alpha)
    };
    let upper = if successes == trials {
        1.0
    } else {
        bisect_probability(|p| binomial_cdf(successes, trials, p), alpha)
    };
    Ok(BinaryInterval {
        successes,
        trials,
        confidence_level,
        lower,
        upper,
    })
}

/// One independent cluster, retaining its stable identity and paired rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterSample {
    /// Stable cluster identity; identities must be unique.
    pub id: u64,
    /// `(baseline, candidate)` rows belonging to this cluster.
    pub pairs: Vec<(f64, f64)>,
}

/// Deterministically bootstraps paired candidate-minus-baseline effects.
pub fn paired_bootstrap_interval(
    pairs: &[(f64, f64)],
    control: BootstrapControl,
) -> StatsResult<BootstrapEffectInterval> {
    control_parts(control)?;
    if pairs.is_empty() {
        return Err(StatsError::EmptyInput {
            metric: "paired bootstrap",
        });
    }
    let effects = pair_effects(pairs, "paired bootstrap")?;
    bootstrap_effects(&effects, control, pairs.len(), pairs.len(), 0)
}

/// Deterministically resamples whole independent clusters with replacement.
///
/// `minimum_clusters` is the caller-declared independence floor. Cluster rows
/// remain together; clusters are sorted by stable id before seeded sampling,
/// making within-cluster row order irrelevant.
pub fn clustered_bootstrap_interval(
    clusters: &[ClusterSample],
    minimum_clusters: usize,
    control: BootstrapControl,
) -> StatsResult<BootstrapEffectInterval> {
    control_parts(control)?;
    if minimum_clusters < 2 {
        return Err(StatsError::InvalidControl {
            field: "minimum_clusters",
            reason: "must be at least two",
        });
    }
    if clusters.len() < minimum_clusters {
        return Err(StatsError::InsufficientInput {
            metric: "clustered bootstrap independent clusters",
            minimum: minimum_clusters,
            actual: clusters.len(),
        });
    }
    let mut ordered = clusters.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|cluster| cluster.id);
    if ordered.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err(StatsError::InvalidControl {
            field: "cluster ids",
            reason: "must be unique",
        });
    }
    let mut cluster_effects = Vec::with_capacity(ordered.len());
    let mut rows = 0usize;
    for cluster in ordered {
        let effects = pair_effects(&cluster.pairs, "clustered bootstrap")?;
        rows = rows
            .checked_add(effects.len())
            .ok_or(StatsError::WorkLimitExceeded {
                required: u64::MAX,
                limit: control.max_work,
            })?;
        cluster_effects.push(effects.iter().sum::<f64>() / effects.len() as f64);
    }
    bootstrap_effects(&cluster_effects, control, rows, rows, clusters.len())
}

/// A pre-registered sequential look and its allocated false-elimination mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegisteredLook {
    /// Cumulative sample count at which this look is legal.
    pub samples: usize,
    /// Positive alpha allocated to this look.
    pub alpha: f64,
}

/// An alpha-spent contract for bounded observations in `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisteredLookSequence {
    looks: Vec<RegisteredLook>,
    total_budget: f64,
}

/// A Hoeffding interval valid at its pre-registered look under the sealed budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialInterval {
    /// Registered cumulative sample count.
    pub samples: usize,
    /// Arithmetic mean of admitted observations.
    pub mean: f64,
    /// Lower endpoint clipped to zero.
    pub lower: f64,
    /// Upper endpoint clipped to one.
    pub upper: f64,
    /// Alpha spent at this look.
    pub alpha_spent: f64,
    /// Caller-declared total false-elimination budget.
    pub total_budget: f64,
}

impl RegisteredLookSequence {
    /// Seals an increasing, unique set of looks whose alpha does not exceed the budget.
    pub fn new(looks: Vec<RegisteredLook>, total_budget: f64) -> StatsResult<Self> {
        if !total_budget.is_finite() || !(0.0..1.0).contains(&total_budget) {
            return Err(StatsError::InvalidControl {
                field: "total_budget",
                reason: "must be finite and strictly between zero and one",
            });
        }
        if looks.is_empty() {
            return Err(StatsError::EmptyInput {
                metric: "registered looks",
            });
        }
        let mut previous = 0;
        let mut spent = 0.0;
        for look in &looks {
            if look.samples == 0 || look.samples <= previous {
                return Err(StatsError::InvalidControl {
                    field: "registered looks",
                    reason: "sample counts must be positive and strictly increasing",
                });
            }
            if !look.alpha.is_finite() || look.alpha <= 0.0 || look.alpha >= 1.0 {
                return Err(StatsError::InvalidControl {
                    field: "look alpha",
                    reason: "must be finite and strictly between zero and one",
                });
            }
            previous = look.samples;
            spent += look.alpha;
        }
        if spent > total_budget + f64::EPSILON * looks.len() as f64 {
            return Err(StatsError::InvalidControl {
                field: "look alpha",
                reason: "sum must not exceed total_budget",
            });
        }
        Ok(Self {
            looks,
            total_budget,
        })
    }

    /// Evaluates exactly one registered look; optional peeking is therefore excluded by construction.
    pub fn interval(&self, observations: &[f64]) -> StatsResult<SequentialInterval> {
        let look = self
            .looks
            .iter()
            .find(|look| look.samples == observations.len())
            .ok_or(StatsError::InvalidControl {
                field: "observations",
                reason: "sample count is not a registered look",
            })?;
        for (index, value) in observations.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(StatsError::NonFinite {
                    metric: "sequential bounded observation",
                    index: Some(index),
                    value: *value,
                });
            }
        }
        let mean = observations.iter().sum::<f64>() / observations.len() as f64;
        let radius = ((2.0 / look.alpha).ln() / (2.0 * observations.len() as f64)).sqrt();
        Ok(SequentialInterval {
            samples: observations.len(),
            mean,
            lower: (mean - radius).max(0.0),
            upper: (mean + radius).min(1.0),
            alpha_spent: look.alpha,
            total_budget: self.total_budget,
        })
    }
}

/// One weighted raw point supplied to isotonic regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotonicPoint {
    /// Tested level, strictly increasing after canonical sorting.
    pub level: f64,
    /// Raw response.
    pub value: f64,
    /// Positive observation weight.
    pub weight: f64,
}

/// Threshold crossing evidence, including censoring beyond the tested range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdReadout {
    /// The first tested level whose fit reaches the threshold.
    Observed {
        /// First tested level reaching the threshold.
        level: f64,
    },
    /// Every fitted point is already at or above the threshold.
    BelowTestedRange,
    /// No fitted point reaches the threshold.
    AboveTestedRange,
}

/// Inspectable weighted pool-adjacent-violators fit.
#[derive(Clone, Debug, PartialEq)]
pub struct IsotonicFit {
    /// Canonically sorted raw points.
    pub raw: Vec<IsotonicPoint>,
    /// Nondecreasing fitted values aligned with `raw`.
    pub fitted: Vec<f64>,
    /// Trapezoidal area divided by the tested level span, available for two or more levels.
    pub normalized_area: Option<f64>,
}

impl IsotonicFit {
    /// Reads a threshold crossing and preserves left/right censoring.
    pub fn threshold(&self, threshold: f64) -> StatsResult<ThresholdReadout> {
        if !threshold.is_finite() {
            return Err(StatsError::NonFinite {
                metric: "isotonic threshold",
                index: None,
                value: threshold,
            });
        }
        if self.fitted[0] >= threshold {
            return Ok(ThresholdReadout::BelowTestedRange);
        }
        Ok(self
            .fitted
            .iter()
            .position(|value| *value >= threshold)
            .map_or(ThresholdReadout::AboveTestedRange, |index| {
                ThresholdReadout::Observed {
                    level: self.raw[index].level,
                }
            }))
    }
}

/// Fits a weighted nondecreasing curve with pool-adjacent-violators.
pub fn fit_isotonic(points: &[IsotonicPoint]) -> StatsResult<IsotonicFit> {
    if points.is_empty() {
        return Err(StatsError::EmptyInput {
            metric: "isotonic points",
        });
    }
    let mut raw = points.to_vec();
    for (index, point) in raw.iter().enumerate() {
        for (metric, value) in [
            ("isotonic level", point.level),
            ("isotonic value", point.value),
            ("isotonic weight", point.weight),
        ] {
            if !value.is_finite() {
                return Err(StatsError::NonFinite {
                    metric,
                    index: Some(index),
                    value,
                });
            }
        }
        if point.weight <= 0.0 {
            return Err(StatsError::InvalidControl {
                field: "isotonic weight",
                reason: "must be positive",
            });
        }
    }
    raw.sort_by(|a, b| a.level.total_cmp(&b.level));
    if raw.windows(2).any(|pair| pair[0].level == pair[1].level) {
        return Err(StatsError::InvalidControl {
            field: "isotonic levels",
            reason: "must be unique",
        });
    }
    let mut blocks: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (index, point) in raw.iter().enumerate() {
        blocks.push((index, index + 1, point.weight, point.weight * point.value));
        while blocks.len() >= 2 {
            let n = blocks.len();
            if blocks[n - 2].3 / blocks[n - 2].2 <= blocks[n - 1].3 / blocks[n - 1].2 {
                break;
            }
            let right = blocks.pop().expect("right block");
            let left = blocks.pop().expect("left block");
            blocks.push((left.0, right.1, left.2 + right.2, left.3 + right.3));
        }
    }
    let mut fitted = vec![0.0; raw.len()];
    for (start, end, weight, sum) in blocks {
        fitted[start..end].fill(sum / weight);
    }
    let normalized_area = (raw.len() >= 2).then(|| {
        let span = raw.last().expect("nonempty").level - raw[0].level;
        raw.windows(2)
            .enumerate()
            .map(|(index, pair)| {
                (pair[1].level - pair[0].level) * (fitted[index] + fitted[index + 1]) / 2.0
            })
            .sum::<f64>()
            / span
    });
    Ok(IsotonicFit {
        raw,
        fitted,
        normalized_area,
    })
}

fn pair_effects(pairs: &[(f64, f64)], metric: &'static str) -> StatsResult<Vec<f64>> {
    if pairs.is_empty() {
        return Err(StatsError::EmptyInput { metric });
    }
    pairs
        .iter()
        .enumerate()
        .map(|(index, (baseline, candidate))| {
            if !baseline.is_finite() {
                return Err(StatsError::NonFinite {
                    metric,
                    index: Some(index * 2),
                    value: *baseline,
                });
            }
            if !candidate.is_finite() {
                return Err(StatsError::NonFinite {
                    metric,
                    index: Some(index * 2 + 1),
                    value: *candidate,
                });
            }
            Ok(candidate - baseline)
        })
        .collect()
}

fn bootstrap_effects(
    effects: &[f64],
    control: BootstrapControl,
    baseline_samples: usize,
    candidate_samples: usize,
    cluster_count: usize,
) -> StatsResult<BootstrapEffectInterval> {
    let required = u64::try_from(effects.len())
        .ok()
        .and_then(|n| n.checked_mul(control.resamples as u64))
        .ok_or(StatsError::WorkLimitExceeded {
            required: u64::MAX,
            limit: control.max_work,
        })?;
    if required > control.max_work {
        return Err(StatsError::WorkLimitExceeded {
            required,
            limit: control.max_work,
        });
    }
    let mut rng = SeededSampler::new(control.seed);
    let mut estimates = Vec::with_capacity(control.resamples);
    for _ in 0..control.resamples {
        estimates.push(
            (0..effects.len())
                .map(|_| effects[rng.index_multiply_high(effects.len())])
                .sum::<f64>()
                / effects.len() as f64,
        );
    }
    let tail = (1.0 - control.confidence_level) / 2.0;
    Ok(BootstrapEffectInterval {
        point_effect: effects.iter().sum::<f64>() / effects.len() as f64,
        lower: exact_quantile(&estimates, tail).map_err(|_| StatsError::InvalidControl {
            field: "bootstrap quantile",
            reason: "internal quantile must remain valid",
        })?,
        upper: exact_quantile(&estimates, 1.0 - tail).map_err(|_| StatsError::InvalidControl {
            field: "bootstrap quantile",
            reason: "internal quantile must remain valid",
        })?,
        confidence_level: control.confidence_level,
        seed: control.seed,
        resamples: control.resamples,
        baseline_samples,
        candidate_samples,
        exclusions: 0,
        cluster_count,
        admitted_work: required,
    })
}

fn control_parts(control: BootstrapControl) -> StatsResult<()> {
    if control.resamples < 2 {
        return Err(StatsError::InvalidControl {
            field: "resamples",
            reason: "must be at least two",
        });
    }
    confidence(control.confidence_level)
}

fn confidence(value: f64) -> StatsResult<()> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(StatsError::InvalidControl {
            field: "confidence_level",
            reason: "must be finite and strictly between zero and one",
        });
    }
    Ok(())
}

fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    (0..=k).map(|i| binomial_probability(i, n, p)).sum()
}
fn binomial_upper_tail(k: u64, n: u64, p: f64) -> f64 {
    (k..=n).map(|i| binomial_probability(i, n, p)).sum()
}
fn binomial_probability(k: u64, n: u64, p: f64) -> f64 {
    if p == 0.0 {
        return f64::from(k == 0);
    }
    if p == 1.0 {
        return f64::from(k == n);
    }
    let log_choose = (1..=k.min(n - k))
        .map(|i| ((n + 1 - i) as f64 / i as f64).ln())
        .sum::<f64>();
    (log_choose + k as f64 * p.ln() + (n - k) as f64 * (-p).ln_1p()).exp()
}
fn bisect_probability(mut tail: impl FnMut(f64) -> f64, target: f64) -> f64 {
    let increasing = tail(0.0) < tail(1.0);
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..80 {
        let mid = (low + high) / 2.0;
        if (tail(mid) < target) == increasing {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) / 2.0
}
