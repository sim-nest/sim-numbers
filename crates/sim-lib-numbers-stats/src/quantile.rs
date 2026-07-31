//! Bounded, mergeable streaming quantiles with an exact small-data path.

use std::{error::Error, fmt, mem};

/// Error and memory policy for a [`QuantileSketch`].
#[derive(Clone, Debug, PartialEq)]
pub struct QuantilePolicy {
    /// Maximum target rank error, as a fraction of the observation count.
    pub rank_error: f64,
    /// Number of observations retained exactly before summarization begins.
    pub exact_threshold: usize,
    /// Hard maximum number of retained summary entries.
    /// Insertion fails instead of silently exceeding this limit or weakening
    /// `rank_error`.
    pub max_summary_entries: usize,
}

impl QuantilePolicy {
    /// Builds a checked policy.
    pub fn new(
        rank_error: f64,
        exact_threshold: usize,
        max_summary_entries: usize,
    ) -> Result<Self, QuantileError> {
        let policy = Self {
            rank_error,
            exact_threshold,
            max_summary_entries,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), QuantileError> {
        if !self.rank_error.is_finite() || !(0.0..0.5).contains(&self.rank_error) {
            return Err(QuantileError::InvalidPolicy {
                field: "rank_error",
                reason: "must be finite and in the half-open interval [0, 0.5)",
            });
        }
        if self.exact_threshold == 0 {
            return Err(QuantileError::InvalidPolicy {
                field: "exact_threshold",
                reason: "must be greater than zero",
            });
        }
        if self.max_summary_entries < self.exact_threshold {
            return Err(QuantileError::InvalidPolicy {
                field: "max_summary_entries",
                reason: "must be at least exact_threshold",
            });
        }
        Ok(())
    }
}

/// One quantile estimate together with its retained rank evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantileEstimate {
    /// Requested quantile in `0.0..=1.0`.
    pub quantile: f64,
    /// Estimated value.
    pub value: f64,
    /// Inclusive lower bound on the value's zero-based normalized rank.
    pub rank_lower: f64,
    /// Inclusive upper bound on the value's zero-based normalized rank.
    pub rank_upper: f64,
    /// Number of observations represented by the sketch.
    pub observations: u64,
    /// Number of retained exact values or summary entries.
    pub retained_entries: usize,
    /// Whether this estimate used the exact small-data reference path.
    pub exact: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SummaryEntry {
    value: f64,
    gap: u64,
    slack: u64,
}

/// A deterministic Greenwald-Khanna streaming quantile summary.
/// The sketch stores observations exactly through
/// [`QuantilePolicy::exact_threshold`]. Larger streams use rank intervals and
/// deterministic compression. Compatible sketches can be merged without
/// replaying their source streams. The hard entry limit makes memory admission
/// explicit: an operation that cannot retain the requested rank error is
/// rejected transactionally.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantileSketch {
    policy: QuantilePolicy,
    observations: u64,
    exact_values: Option<Vec<f64>>,
    summary: Vec<SummaryEntry>,
}

impl QuantileSketch {
    /// Creates an empty sketch governed by `policy`.
    pub fn new(policy: QuantilePolicy) -> Result<Self, QuantileError> {
        policy.validate()?;
        Ok(Self {
            policy,
            observations: 0,
            exact_values: Some(Vec::new()),
            summary: Vec::new(),
        })
    }

    /// Returns the immutable error and memory policy.
    pub fn policy(&self) -> &QuantilePolicy {
        &self.policy
    }

    /// Returns the number of observations represented by the sketch.
    pub fn len(&self) -> u64 {
        self.observations
    }

    /// Returns whether the sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Returns the number of retained exact values or summary entries.
    pub fn retained_entries(&self) -> usize {
        self.exact_values
            .as_ref()
            .map_or(self.summary.len(), Vec::len)
    }

    /// Returns an upper bound on bytes occupied by retained numeric entries.
    /// This excludes allocator metadata and the fixed-size sketch value.
    pub fn retained_entry_bytes(&self) -> usize {
        if let Some(values) = &self.exact_values {
            values.len() * mem::size_of::<f64>()
        } else {
            self.summary.len() * mem::size_of::<SummaryEntry>()
        }
    }

    /// Inserts one finite observation.
    /// The operation is transactional when the configured entry limit is too
    /// small for the requested error.
    pub fn insert(&mut self, value: f64) -> Result<(), QuantileError> {
        if !value.is_finite() {
            return Err(QuantileError::NonFinite { index: None, value });
        }
        let mut candidate = self.clone();
        candidate.insert_unchecked(value)?;
        *self = candidate;
        Ok(())
    }

    /// Merges another compatible sketch transactionally.
    pub fn merge(&mut self, other: &Self) -> Result<(), QuantileError> {
        if self.policy != other.policy {
            return Err(QuantileError::IncompatiblePolicy);
        }
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            *self = other.clone();
            return Ok(());
        }
        let mut candidate = self.clone();
        candidate.merge_unchecked(other)?;
        *self = candidate;
        Ok(())
    }

    /// Estimates a quantile and returns its retained rank interval.
    pub fn estimate(&self, quantile: f64) -> Result<QuantileEstimate, QuantileError> {
        validate_quantile(quantile)?;
        if self.is_empty() {
            return Err(QuantileError::EmptyInput);
        }
        if let Some(values) = &self.exact_values {
            let value = exact_quantile(values, quantile)?;
            let normalized_rank = if values.len() == 1 { 0.0 } else { quantile };
            return Ok(QuantileEstimate {
                quantile,
                value,
                rank_lower: normalized_rank,
                rank_upper: normalized_rank,
                observations: self.observations,
                retained_entries: values.len(),
                exact: true,
            });
        }

        let last = self.summary.len() - 1;
        let selected = if quantile == 0.0 {
            0
        } else if quantile == 1.0 {
            last
        } else {
            self.summary_index(quantile)
        };
        let (rank_lower, rank_upper) = self.rank_interval(selected);
        Ok(QuantileEstimate {
            quantile,
            value: self.summary[selected].value,
            rank_lower,
            rank_upper,
            observations: self.observations,
            retained_entries: self.summary.len(),
            exact: false,
        })
    }

    fn insert_unchecked(&mut self, value: f64) -> Result<(), QuantileError> {
        if let Some(values) = &mut self.exact_values {
            values.push(value);
            self.observations = self
                .observations
                .checked_add(1)
                .ok_or(QuantileError::CountOverflow)?;
            if values.len() <= self.policy.exact_threshold {
                return Ok(());
            }
            let exact = self.exact_values.take().unwrap_or_default();
            self.observations = 0;
            for exact_value in exact {
                self.insert_summary_value(exact_value)?;
            }
            return Ok(());
        }
        self.insert_summary_value(value)
    }

    fn insert_summary_value(&mut self, value: f64) -> Result<(), QuantileError> {
        self.observations = self
            .observations
            .checked_add(1)
            .ok_or(QuantileError::CountOverflow)?;
        let position = self.summary.partition_point(|entry| entry.value <= value);
        let slack = if position == 0 || position == self.summary.len() {
            0
        } else {
            self.allowable_error().saturating_sub(1)
        };
        self.summary.insert(
            position,
            SummaryEntry {
                value,
                gap: 1,
                slack,
            },
        );
        self.compress()?;
        self.check_memory()
    }

    fn merge_unchecked(&mut self, other: &Self) -> Result<(), QuantileError> {
        let combined = self
            .observations
            .checked_add(other.observations)
            .ok_or(QuantileError::CountOverflow)?;
        if combined <= self.policy.exact_threshold as u64 {
            let mut values = self.exact_values.take().unwrap_or_default();
            values.extend(other.exact_values.as_deref().unwrap_or_default());
            self.observations = combined;
            self.exact_values = Some(values);
            self.summary.clear();
            return Ok(());
        }

        let own_entries = self.entries_for_merge();
        let other_entries = other.entries_for_merge();
        let mut merged = Vec::with_capacity(own_entries.len() + other_entries.len());
        for (entries, other_entries) in [
            (own_entries.as_slice(), other_entries.as_slice()),
            (other_entries.as_slice(), own_entries.as_slice()),
        ] {
            for entry in entries.iter().copied() {
                merged.push(SummaryEntry {
                    slack: entry
                        .slack
                        .saturating_add(cross_rank_uncertainty(entry.value, other_entries)),
                    ..entry
                });
            }
        }
        merged.sort_by(|left, right| left.value.total_cmp(&right.value));
        if let Some(first) = merged.first_mut() {
            first.slack = 0;
        }
        if let Some(last) = merged.last_mut() {
            last.slack = 0;
        }
        self.observations = combined;
        self.exact_values = None;
        self.summary = merged;
        self.compress()?;
        self.check_memory()
    }

    fn entries_for_merge(&self) -> Vec<SummaryEntry> {
        self.exact_values.as_ref().map_or_else(
            || self.summary.clone(),
            |values| {
                let mut values = values.clone();
                values.sort_by(f64::total_cmp);
                values
                    .into_iter()
                    .map(|value| SummaryEntry {
                        value,
                        gap: 1,
                        slack: 0,
                    })
                    .collect()
            },
        )
    }

    fn compress(&mut self) -> Result<(), QuantileError> {
        if self.summary.len() < 3 {
            return Ok(());
        }
        let allowable = self.allowable_error();
        let mut index = self.summary.len() - 2;
        while index > 0 {
            let combined = self.summary[index]
                .gap
                .checked_add(self.summary[index + 1].gap)
                .and_then(|value| value.checked_add(self.summary[index + 1].slack))
                .ok_or(QuantileError::CountOverflow)?;
            if combined <= allowable {
                let removed = self.summary.remove(index);
                self.summary[index].gap = self.summary[index]
                    .gap
                    .checked_add(removed.gap)
                    .ok_or(QuantileError::CountOverflow)?;
            }
            index -= 1;
        }
        Ok(())
    }

    fn summary_index(&self, quantile: f64) -> usize {
        let target = quantile.mul_add(self.observations.saturating_sub(1) as f64, 1.0);
        let permitted = self.policy.rank_error * self.observations as f64;
        let mut minimum_rank = 0_u64;
        let mut previous = 0;
        for (index, entry) in self.summary.iter().enumerate() {
            minimum_rank = minimum_rank.saturating_add(entry.gap);
            let maximum_rank = minimum_rank.saturating_add(entry.slack);
            if maximum_rank as f64 > target + permitted {
                return previous;
            }
            previous = index;
        }
        self.summary.len() - 1
    }

    fn rank_interval(&self, selected: usize) -> (f64, f64) {
        if self.observations <= 1 {
            return (0.0, 0.0);
        }
        let minimum = self.summary[..=selected]
            .iter()
            .map(|entry| entry.gap)
            .sum::<u64>();
        let maximum = minimum.saturating_add(self.summary[selected].slack);
        let denominator = (self.observations - 1) as f64;
        (
            minimum.saturating_sub(1) as f64 / denominator,
            maximum.saturating_sub(1) as f64 / denominator,
        )
    }

    fn allowable_error(&self) -> u64 {
        allowance(self.policy.rank_error, self.observations)
    }

    fn check_memory(&self) -> Result<(), QuantileError> {
        if self.summary.len() > self.policy.max_summary_entries {
            return Err(QuantileError::MemoryLimit {
                required_entries: self.summary.len(),
                max_entries: self.policy.max_summary_entries,
            });
        }
        Ok(())
    }
}

/// Computes the exact linearly interpolated quantile of finite small data.
pub fn exact_quantile(values: &[f64], quantile: f64) -> Result<f64, QuantileError> {
    validate_quantile(quantile)?;
    if values.is_empty() {
        return Err(QuantileError::EmptyInput);
    }
    let mut ordered = values.to_vec();
    for (index, value) in ordered.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(QuantileError::NonFinite {
                index: Some(index),
                value,
            });
        }
    }
    ordered.sort_by(f64::total_cmp);
    let position = quantile * (ordered.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Ok((ordered[upper] - ordered[lower]).mul_add(fraction, ordered[lower]))
}

/// Failure while configuring, updating, merging, or querying a quantile sketch.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantileError {
    /// No observations were supplied.
    EmptyInput,
    /// A policy field was invalid.
    InvalidPolicy {
        /// Invalid field name.
        field: &'static str,
        /// Concrete policy requirement.
        reason: &'static str,
    },
    /// A quantile was outside `0.0..=1.0` or not finite.
    QuantileOutOfRange {
        /// Rejected quantile.
        quantile: f64,
    },
    /// An observation was not finite.
    NonFinite {
        /// Position in an exact input slice, when available.
        index: Option<usize>,
        /// Rejected value.
        value: f64,
    },
    /// Two sketches had different error or memory policies.
    IncompatiblePolicy,
    /// The configured entry bound could not preserve the requested rank error.
    MemoryLimit {
        /// Entries required after compression.
        required_entries: usize,
        /// Configured hard entry bound.
        max_entries: usize,
    },
    /// The represented observation count exceeded `u64`.
    CountOverflow,
}

impl fmt::Display for QuantileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "quantile input must not be empty"),
            Self::InvalidPolicy { field, reason } => {
                write!(formatter, "invalid quantile policy {field}: {reason}")
            }
            Self::QuantileOutOfRange { quantile } => {
                write!(formatter, "quantile outside finite 0..=1 range: {quantile}")
            }
            Self::NonFinite { index, value } => match index {
                Some(index) => write!(formatter, "quantile value {index} is not finite: {value}"),
                None => write!(formatter, "quantile observation is not finite: {value}"),
            },
            Self::IncompatiblePolicy => write!(formatter, "quantile policies do not match"),
            Self::MemoryLimit {
                required_entries,
                max_entries,
            } => write!(
                formatter,
                "quantile summary requires {required_entries} entries, limit is {max_entries}"
            ),
            Self::CountOverflow => write!(formatter, "quantile observation count overflow"),
        }
    }
}

impl Error for QuantileError {}

fn validate_quantile(quantile: f64) -> Result<(), QuantileError> {
    if quantile.is_finite() && (0.0..=1.0).contains(&quantile) {
        Ok(())
    } else {
        Err(QuantileError::QuantileOutOfRange { quantile })
    }
}

fn allowance(rank_error: f64, observations: u64) -> u64 {
    (2.0 * rank_error * observations as f64).floor() as u64
}

fn cross_rank_uncertainty(value: f64, entries: &[SummaryEntry]) -> u64 {
    let upper = entries.partition_point(|entry| entry.value <= value);
    if upper == 0 || upper == entries.len() {
        0
    } else {
        entries[upper]
            .gap
            .saturating_add(entries[upper].slack)
            .saturating_sub(1)
    }
}
