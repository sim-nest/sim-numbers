//! Versioned deterministic sampling and bounded space-filling designs.

use core::fmt;

/// Stable identity of a deterministic sampler algorithm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerAlgorithm {
    /// SplitMix64 with the published Steele-Vigna mixing constants, version 1.
    SplitMix64V1,
}

/// Canonical, architecture-independent sampler state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplerState {
    /// Algorithm and stream-version identity.
    pub algorithm: SamplerAlgorithm,
    /// Current 64-bit state word.
    pub state: u64,
    /// Exact number of generated 64-bit words.
    pub words_generated: u64,
    /// Root seed from which this stream was constructed.
    pub seed: u64,
    /// Substream label, zero for the root stream.
    pub stream: u64,
}

/// Replay receipt captured at an observable sampling boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplerReceipt {
    /// State sufficient to resume the stream exactly.
    pub state: SamplerState,
    /// Optional hard word allowance selected by the caller.
    pub max_words: Option<u64>,
}

/// A public deterministic sampler with explicit version, work, state and forks.
///
/// `fork(label)` is domain-separated from both the root stream and other labels;
/// it never consumes its parent. Existing consumers use the root stream and thus
/// retain their historical SplitMix64-v1 output exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SeededSampler {
    state: SamplerState,
    max_words: Option<u64>,
}

impl SeededSampler {
    /// Constructs an unbounded root stream. Algorithms should preflight their
    /// own exact work or use [`Self::with_max_words`] for refusal.
    pub fn new(seed: u64) -> Self {
        Self::from_parts(seed, 0, seed, None)
    }

    /// Constructs a root stream that refuses generation beyond `max_words`.
    pub fn with_max_words(seed: u64, max_words: u64) -> Self {
        Self::from_parts(seed, 0, seed, Some(max_words))
    }

    fn from_parts(seed: u64, stream: u64, state: u64, max_words: Option<u64>) -> Self {
        Self {
            state: SamplerState {
                algorithm: SamplerAlgorithm::SplitMix64V1,
                state,
                words_generated: 0,
                seed,
                stream,
            },
            max_words,
        }
    }

    /// Restores a canonical state and word allowance.
    pub fn replay(receipt: SamplerReceipt) -> Self {
        Self {
            state: receipt.state,
            max_words: receipt.max_words,
        }
    }

    /// Captures a replay receipt.
    pub fn receipt(&self) -> SamplerReceipt {
        SamplerReceipt {
            state: self.state,
            max_words: self.max_words,
        }
    }

    /// Returns an independent, deterministic labeled substream without
    /// advancing this stream.
    pub fn fork(&self, label: u64) -> Self {
        let state = mix(self.state.seed
            ^ label.wrapping_mul(0xd2b7_4407_b1ce_6e93)
            ^ 0xa076_1d64_78bd_642f);
        Self::from_parts(self.state.seed, label, state, self.max_words)
    }

    /// Produces the next word, refusing when the declared allowance is spent.
    pub fn try_next_u64(&mut self) -> Result<u64, DesignError> {
        if self
            .max_words
            .is_some_and(|limit| self.state.words_generated >= limit)
        {
            return Err(DesignError::WorkLimit {
                required: self.state.words_generated.saturating_add(1),
                limit: self.max_words.unwrap_or(0),
            });
        }
        self.state.state = self.state.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.state.words_generated += 1;
        Ok(mix(self.state.state))
    }

    /// Produces the next word on a stream whose work was preflighted.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.try_next_u64()
            .expect("unbounded or preflighted sampler")
    }

    /// Maps one word to `[0, 1)` using the stable 53-bit convention.
    pub fn unit_interval(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Historical modulo mapping used by clustering and HMM fixtures.
    pub(crate) fn index_modulo(&mut self, length: usize) -> usize {
        (self.next_u64() % length as u64) as usize
    }

    /// Historical multiply-high mapping used by bootstrap fixtures.
    pub(crate) fn index_multiply_high(&mut self, length: usize) -> usize {
        ((u128::from(self.next_u64()) * length as u128) >> 64) as usize
    }
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Optional reviewed digital scrambling policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scramble {
    /// Preserve the canonical digital net.
    None,
    /// Apply a deterministic digital XOR shift from the sampler seed.
    DigitalShift,
}

/// Caller-described region intentionally absent from a sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct UntestedRegion {
    /// Stable caller-owned label.
    pub label: String,
    /// Human-reviewable reason for exclusion.
    pub reason: String,
}

/// Coverage and replay evidence returned with every design.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageEvidence {
    /// Sequence identity including algorithm version and policy.
    pub sequence_identity: String,
    /// Exact zero-based boundary point indices injected by policy.
    pub boundary_injections: Vec<usize>,
    /// Pairs `(later, earlier)` for exactly duplicated points.
    pub duplicates: Vec<(usize, usize)>,
    /// Per-dimension occupancy counts for every declared stratum.
    pub stratum_occupancy: Vec<Vec<usize>>,
    /// Final sampler state when a sampler participated.
    pub sampler: Option<SamplerReceipt>,
    /// Caller-supplied excluded-region metadata, preserved verbatim.
    pub untested_regions: Vec<UntestedRegion>,
    /// Exact admitted work units.
    pub work: u64,
}

/// A generated, reconstructable design and its coverage evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleDesign {
    /// Row-major points in the unit hypercube.
    pub points: Vec<Vec<f64>>,
    /// Auditable coverage and replay facts.
    pub coverage: CoverageEvidence,
}

/// Latin hypercube request.
#[derive(Clone, Debug, PartialEq)]
pub struct LatinHypercubePlan {
    /// Number of coordinates, bounded to 64.
    pub dimensions: usize,
    /// Number of points and strata.
    pub points: usize,
    /// Root seed.
    pub seed: u64,
    /// Maximum shuffle draws.
    pub max_work: u64,
    /// Caller-owned exclusions.
    pub untested_regions: Vec<UntestedRegion>,
}

impl LatinHypercubePlan {
    /// Generates a centered Latin hypercube with one point per stratum in every dimension.
    pub fn generate(&self) -> Result<SampleDesign, DesignError> {
        validate_shape(self.dimensions, self.points, 64)?;
        let required = (self.dimensions as u64)
            .checked_mul(self.points.saturating_sub(1) as u64)
            .ok_or(DesignError::Overflow)?;
        if required > self.max_work {
            return Err(DesignError::WorkLimit {
                required,
                limit: self.max_work,
            });
        }
        let mut sampler = SeededSampler::with_max_words(self.seed, self.max_work);
        let mut points = vec![vec![0.0; self.dimensions]; self.points];
        let occupancy = vec![vec![1; self.points]; self.dimensions];
        for (dimension, _) in occupancy.iter().enumerate() {
            let mut permutation = (0..self.points).collect::<Vec<_>>();
            for end in (1..self.points).rev() {
                let chosen = sampler.index_modulo(end + 1);
                permutation.swap(end, chosen);
            }
            for (row, stratum) in permutation.into_iter().enumerate() {
                points[row][dimension] = (stratum as f64 + 0.5) / self.points as f64;
            }
        }
        Ok(design(
            points,
            format!("latin-hypercube/centered-v1;seed={}", self.seed),
            vec![],
            occupancy,
            Some(sampler.receipt()),
            self.untested_regions.clone(),
            required,
        ))
    }
}

/// Sobol base-2 digital-net request. The reviewed direction table is bounded to four dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct SobolPlan {
    /// Dimensions in `1..=4`.
    pub dimensions: usize,
    /// Point count.
    pub points: usize,
    /// Number of canonical prefix points to skip.
    pub skip: u64,
    /// Scrambling policy.
    pub scramble: Scramble,
    /// Seed used only by reviewed scrambling.
    pub seed: u64,
    /// Maximum coordinate work.
    pub max_work: u64,
    /// Caller-owned exclusions.
    pub untested_regions: Vec<UntestedRegion>,
}

impl SobolPlan {
    /// Generates the requested bounded Sobol prefix.
    pub fn generate(&self) -> Result<SampleDesign, DesignError> {
        validate_shape(self.dimensions, self.points, 4)?;
        let end = self
            .skip
            .checked_add(self.points as u64)
            .ok_or(DesignError::Overflow)?;
        if end > u32::MAX as u64 {
            return Err(DesignError::UnsupportedPoint { point: end });
        }
        let required = (self.dimensions as u64)
            .checked_mul(self.points as u64)
            .ok_or(DesignError::Overflow)?;
        if required > self.max_work {
            return Err(DesignError::WorkLimit {
                required,
                limit: self.max_work,
            });
        }
        let mut sampler = SeededSampler::new(self.seed);
        let shifts = (0..self.dimensions)
            .map(|_| {
                if self.scramble == Scramble::DigitalShift {
                    sampler.next_u64()
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let mut points = Vec::with_capacity(self.points);
        for index in self.skip..end {
            let gray = index ^ (index >> 1);
            let mut row = Vec::with_capacity(self.dimensions);
            for (dimension, shift) in shifts.iter().copied().enumerate() {
                let mut bits = 0_u64;
                for bit in 0..32 {
                    if gray & (1_u64 << bit) != 0 {
                        bits ^= direction(dimension, bit);
                    }
                }
                row.push(((bits ^ shift) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64)));
            }
            points.push(row);
        }
        let receipt = (self.scramble == Scramble::DigitalShift).then(|| sampler.receipt());
        Ok(design(
            points,
            format!(
                "sobol/joe-kuo-reviewed-4d-v1;skip={};scramble={:?};seed={}",
                self.skip, self.scramble, self.seed
            ),
            vec![],
            vec![],
            receipt,
            self.untested_regions.clone(),
            required,
        ))
    }
}

// Bratley-Fox direction recurrence for dimensions 1..4: (s, a, m_i).
fn direction(dimension: usize, bit: usize) -> u64 {
    if dimension == 0 {
        return 1_u64 << (63 - bit);
    }
    let (s, a, initial): (usize, u32, &[u32]) = match dimension {
        1 => (1, 0, &[1]),
        2 => (2, 1, &[1, 3]),
        3 => (3, 1, &[1, 3, 1]),
        _ => unreachable!(),
    };
    let mut values = [0_u64; 32];
    for index in 0..s {
        values[index] = (initial[index] as u64) << (63 - index);
    }
    for index in s..=bit {
        let mut value = values[index - s] ^ (values[index - s] >> s);
        for k in 1..s {
            if ((a >> (s - 1 - k)) & 1) != 0 {
                value ^= values[index - k];
            }
        }
        values[index] = value;
    }
    values[bit]
}

/// Boundary-injection wrapper for any already generated unit-cube design.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepPlan {
    /// Whether to prepend the all-zero corner.
    pub inject_lower_boundary: bool,
    /// Whether to append the all-one corner.
    pub inject_upper_boundary: bool,
    /// Caller-owned exclusions.
    pub untested_regions: Vec<UntestedRegion>,
}

impl SweepPlan {
    /// Injects requested exact boundaries and recomputes duplicate evidence.
    pub fn apply(&self, mut design: SampleDesign) -> SampleDesign {
        let dimensions = design.points.first().map_or(0, Vec::len);
        let mut boundaries = Vec::new();
        if self.inject_lower_boundary {
            design.points.insert(0, vec![0.0; dimensions]);
            boundaries.push(0);
        }
        if self.inject_upper_boundary {
            boundaries.push(design.points.len());
            design.points.push(vec![1.0; dimensions]);
        }
        design.coverage.boundary_injections = boundaries;
        design.coverage.duplicates = duplicates(&design.points);
        design
            .coverage
            .untested_regions
            .extend(self.untested_regions.clone());
        design
            .coverage
            .sequence_identity
            .push_str(";sweep-boundaries-v1");
        design
    }
}

fn validate_shape(
    dimensions: usize,
    points: usize,
    max_dimensions: usize,
) -> Result<(), DesignError> {
    if dimensions == 0 || dimensions > max_dimensions {
        return Err(DesignError::UnsupportedDimension {
            requested: dimensions,
            maximum: max_dimensions,
        });
    }
    if points == 0 {
        return Err(DesignError::InvalidPointCount);
    }
    Ok(())
}
fn design(
    points: Vec<Vec<f64>>,
    identity: String,
    boundaries: Vec<usize>,
    occupancy: Vec<Vec<usize>>,
    sampler: Option<SamplerReceipt>,
    untested_regions: Vec<UntestedRegion>,
    work: u64,
) -> SampleDesign {
    let duplicates = duplicates(&points);
    SampleDesign {
        points,
        coverage: CoverageEvidence {
            sequence_identity: identity,
            boundary_injections: boundaries,
            duplicates,
            stratum_occupancy: occupancy,
            sampler,
            untested_regions,
            work,
        },
    }
}
fn duplicates(points: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for later in 0..points.len() {
        if let Some(earlier) = (0..later).find(|&earlier| points[earlier] == points[later]) {
            result.push((later, earlier));
        }
    }
    result
}

/// Fail-closed design and sampler errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DesignError {
    /// Dimension exceeds the reviewed table or is zero.
    UnsupportedDimension {
        /// Requested dimension.
        requested: usize,
        /// Maximum reviewed dimension.
        maximum: usize,
    },
    /// Zero points were requested.
    InvalidPointCount,
    /// Skip plus point count exceeds the supported prefix.
    UnsupportedPoint {
        /// First unsupported point index.
        point: u64,
    },
    /// Exact work exceeds policy.
    WorkLimit {
        /// Required work.
        required: u64,
        /// Allowed work.
        limit: u64,
    },
    /// Size arithmetic overflowed.
    Overflow,
}
impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for DesignError {}
