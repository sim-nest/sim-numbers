//! Finite, inspectable first-order Markov transition estimation.

use super::transition::FiniteTransitionMatrix;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// Stable provenance attached to every fitted transition model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusProvenance {
    /// Stable corpus identifier.
    pub id: String,
    /// Human-readable origin, generator, or public source.
    pub source: String,
    /// SPDX license identifier or an equally precise public-domain declaration.
    pub license: String,
    /// Content hash of the exact corpus bytes.
    pub content_hash: String,
}

impl CorpusProvenance {
    /// Builds checked provenance from a previously computed content hash.
    pub fn new(
        id: impl Into<String>,
        source: impl Into<String>,
        license: impl Into<String>,
        content_hash: impl Into<String>,
    ) -> Result<Self, MarkovError> {
        let provenance = Self {
            id: id.into(),
            source: source.into(),
            license: license.into(),
            content_hash: content_hash.into(),
        };
        provenance.validate()?;
        Ok(provenance)
    }

    /// Builds provenance and computes an FNV-1a hash over the exact corpus bytes.
    pub fn from_bytes(
        id: impl Into<String>,
        source: impl Into<String>,
        license: impl Into<String>,
        bytes: &[u8],
    ) -> Result<Self, MarkovError> {
        Self::new(id, source, license, fnv1a64(bytes))
    }

    fn validate(&self) -> Result<(), MarkovError> {
        for (field, value) in [
            ("corpus.id", self.id.as_str()),
            ("corpus.source", self.source.as_str()),
            ("corpus.license", self.license.as_str()),
            ("corpus.content_hash", self.content_hash.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MarkovError::InvalidPolicy {
                    field,
                    reason: "must not be empty",
                });
            }
        }
        Ok(())
    }
}

/// Explicit fitting and evaluation policy for a finite first-order model.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovPolicy {
    /// Additive pseudo-count applied to every transition in the finite state set.
    pub additive_smoothing: f64,
    /// Number of trailing sequences reserved for held-out evaluation.
    pub held_out_sequences: usize,
    /// Identity, origin, license, and exact content hash of the corpus.
    pub corpus: CorpusProvenance,
}

impl MarkovPolicy {
    /// Builds a policy and rejects absent smoothing or incomplete provenance.
    pub fn new(
        additive_smoothing: f64,
        held_out_sequences: usize,
        corpus: CorpusProvenance,
    ) -> Result<Self, MarkovError> {
        let policy = Self {
            additive_smoothing,
            held_out_sequences,
            corpus,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), MarkovError> {
        if !self.additive_smoothing.is_finite() || self.additive_smoothing <= 0.0 {
            return Err(MarkovError::InvalidPolicy {
                field: "additive_smoothing",
                reason: "must be finite and greater than zero",
            });
        }
        self.corpus.validate()
    }
}

/// Aggregate likelihood evidence for a collection of state sequences.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionScore {
    /// Number of adjacent transitions scored.
    pub transitions: u64,
    /// Sum of natural logarithms of smoothed transition probabilities.
    pub log_likelihood: f64,
    /// Mean negative natural-log likelihood per transition.
    pub mean_negative_log_likelihood: f64,
    /// Exponential of the mean negative log likelihood.
    pub perplexity: f64,
}

/// A fitted value together with training and held-out evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelReport<M> {
    /// Inspectable fitted model.
    pub model: M,
    /// Number of sequences used to estimate transition counts.
    pub training_sequences: usize,
    /// Number of trailing sequences excluded from estimation.
    pub held_out_sequences: usize,
    /// Likelihood of the exact training partition under the fitted model.
    pub training_score: TransitionScore,
    /// Likelihood of the held-out partition, when one was requested.
    pub held_out_score: Option<TransitionScore>,
}

/// A finite first-order Markov model retaining exact counts and fitting policy.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovModel<S> {
    states: Vec<S>,
    transition_counts: BTreeMap<(S, S), u64>,
    outgoing_counts: BTreeMap<S, u64>,
    policy: MarkovPolicy,
}

impl<S: Ord + Clone> MarkovModel<S> {
    /// Returns the sorted finite state vocabulary.
    pub fn states(&self) -> &[S] {
        &self.states
    }

    /// Returns the policy and corpus provenance used during fitting.
    pub fn policy(&self) -> &MarkovPolicy {
        &self.policy
    }

    /// Projects the fitted counts and smoothing policy into the shared finite
    /// transition representation used by hidden-state inference.
    pub fn transition_matrix(&self) -> FiniteTransitionMatrix<S> {
        let probabilities = self
            .states
            .iter()
            .map(|from| {
                self.states
                    .iter()
                    .map(|to| {
                        let count = self
                            .transition_counts
                            .get(&(from.clone(), to.clone()))
                            .copied()
                            .unwrap_or(0) as f64;
                        let outgoing = self.outgoing_counts.get(from).copied().unwrap_or(0) as f64;
                        let smoothing = self.policy.additive_smoothing;
                        (count + smoothing) / (outgoing + smoothing * self.states.len() as f64)
                    })
                    .collect()
            })
            .collect();
        FiniteTransitionMatrix::from_normalized(self.states.clone(), probabilities)
    }

    /// Returns the exact observed count for one transition.
    pub fn transition_count(&self, from: &S, to: &S) -> Result<u64, MarkovError> {
        self.require_state(from, 0, 0)?;
        self.require_state(to, 0, 1)?;
        Ok(self
            .transition_counts
            .get(&(from.clone(), to.clone()))
            .copied()
            .unwrap_or(0))
    }

    /// Returns the additively smoothed probability of one transition.
    pub fn transition_probability(&self, from: &S, to: &S) -> Result<f64, MarkovError> {
        let count = self.transition_count(from, to)? as f64;
        let outgoing = self.outgoing_counts.get(from).copied().unwrap_or(0) as f64;
        let smoothing = self.policy.additive_smoothing;
        Ok((count + smoothing) / (outgoing + smoothing * self.states.len() as f64))
    }

    /// Scores sequences without changing the fitted model.
    pub fn score(&self, sequences: &[Vec<S>]) -> Result<TransitionScore, MarkovError> {
        score_sequences(self, sequences, "evaluation")
    }

    /// Serializes policy, provenance, states, and exact counts deterministically.
    ///
    /// The caller supplies a stable, domain-owned state label. Labels are
    /// hex-encoded and indexed in sorted state order, so punctuation and
    /// whitespace cannot make the representation ambiguous.
    pub fn to_stable_text(
        &self,
        mut state_label: impl FnMut(&S) -> String,
    ) -> Result<String, MarkovError> {
        let labels = self.states.iter().map(&mut state_label).collect::<Vec<_>>();
        let unique = labels.iter().collect::<BTreeSet<_>>();
        if unique.len() != labels.len() {
            return Err(MarkovError::DuplicateStateLabel);
        }
        let indices = self
            .states
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, state)| (state, index))
            .collect::<BTreeMap<_, _>>();
        let mut text = String::from("SIM-MARKOV-1\n");
        text.push_str(&format!(
            "additive-smoothing-bits={:016x}\n",
            self.policy.additive_smoothing.to_bits()
        ));
        text.push_str(&format!(
            "held-out-sequences={}\n",
            self.policy.held_out_sequences
        ));
        for (name, value) in [
            ("corpus-id", self.policy.corpus.id.as_str()),
            ("corpus-source", self.policy.corpus.source.as_str()),
            ("corpus-license", self.policy.corpus.license.as_str()),
            ("corpus-hash", self.policy.corpus.content_hash.as_str()),
        ] {
            text.push_str(name);
            text.push('=');
            text.push_str(&hex(value.as_bytes()));
            text.push('\n');
        }
        text.push_str(&format!("states={}\n", labels.len()));
        for (index, label) in labels.iter().enumerate() {
            text.push_str(&format!("state={index}:{}\n", hex(label.as_bytes())));
        }
        for ((from, to), count) in &self.transition_counts {
            text.push_str(&format!(
                "transition={}:{}:{count}\n",
                indices[from], indices[to]
            ));
        }
        Ok(text)
    }

    fn require_state(
        &self,
        state: &S,
        sequence: usize,
        position: usize,
    ) -> Result<(), MarkovError> {
        if self.states.binary_search(state).is_err() {
            return Err(MarkovError::UnknownState { sequence, position });
        }
        Ok(())
    }
}

/// Fits an inspectable finite first-order model and reports held-out evidence.
///
/// The last `policy.held_out_sequences` are excluded from both the finite
/// vocabulary and count estimation. A held-out state absent from training
/// therefore fails closed instead of leaking evaluation data into the model.
pub fn fit_markov<S: Ord + Clone>(
    sequences: &[Vec<S>],
    policy: MarkovPolicy,
) -> Result<ModelReport<MarkovModel<S>>, MarkovError> {
    policy.validate()?;
    if sequences.is_empty() {
        return Err(MarkovError::EmptyCorpus);
    }
    if policy.held_out_sequences >= sequences.len() {
        return Err(MarkovError::InvalidHoldout {
            sequences: sequences.len(),
            held_out: policy.held_out_sequences,
        });
    }
    for (index, sequence) in sequences.iter().enumerate() {
        if sequence.is_empty() {
            return Err(MarkovError::EmptySequence { index });
        }
    }

    let split = sequences.len() - policy.held_out_sequences;
    require_transitions(&sequences[..split], "training")?;
    if split < sequences.len() {
        require_transitions(&sequences[split..], "held-out")?;
    }
    let states = sequences[..split]
        .iter()
        .flat_map(|sequence| sequence.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut transition_counts = BTreeMap::new();
    let mut outgoing_counts = BTreeMap::new();
    for sequence in &sequences[..split] {
        for pair in sequence.windows(2) {
            increment(
                transition_counts
                    .entry((pair[0].clone(), pair[1].clone()))
                    .or_insert(0),
            )?;
            increment(outgoing_counts.entry(pair[0].clone()).or_insert(0))?;
        }
    }
    let model = MarkovModel {
        states,
        transition_counts,
        outgoing_counts,
        policy,
    };
    let training_score = score_sequences(&model, &sequences[..split], "training")?;
    let held_out_score = if split < sequences.len() {
        Some(score_sequences(&model, &sequences[split..], "held-out")?)
    } else {
        None
    };
    Ok(ModelReport {
        model,
        training_sequences: split,
        held_out_sequences: sequences.len() - split,
        training_score,
        held_out_score,
    })
}

/// Computes a stable FNV-1a digest for small transparent fixture corpora.
pub fn fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

/// Failure while validating, fitting, scoring, or serializing a Markov model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkovError {
    /// The corpus contained no sequences.
    EmptyCorpus,
    /// One sequence contained no state.
    EmptySequence {
        /// Zero-based sequence position.
        index: usize,
    },
    /// The requested partition contained no adjacent state pair.
    NoTransitions {
        /// Stable partition name.
        partition: &'static str,
    },
    /// The holdout would leave no training sequence.
    InvalidHoldout {
        /// Total sequence count.
        sequences: usize,
        /// Requested held-out count.
        held_out: usize,
    },
    /// Policy or provenance was incomplete or numerically invalid.
    InvalidPolicy {
        /// Invalid field.
        field: &'static str,
        /// Concrete requirement.
        reason: &'static str,
    },
    /// A scored state was outside the fitted finite vocabulary.
    UnknownState {
        /// Zero-based sequence position.
        sequence: usize,
        /// Zero-based state position.
        position: usize,
    },
    /// A transition count exceeded `u64`.
    CountOverflow,
    /// Two distinct states were given the same stable serialization label.
    DuplicateStateLabel,
}

impl fmt::Display for MarkovError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCorpus => write!(formatter, "Markov fitting requires at least one sequence"),
            Self::EmptySequence { index } => {
                write!(formatter, "Markov sequence {index} contains no state")
            }
            Self::NoTransitions { partition } => {
                write!(
                    formatter,
                    "Markov {partition} partition contains no transition"
                )
            }
            Self::InvalidHoldout {
                sequences,
                held_out,
            } => write!(
                formatter,
                "Markov holdout {held_out} must be smaller than sequence count {sequences}"
            ),
            Self::InvalidPolicy { field, reason } => {
                write!(formatter, "invalid Markov policy {field}: {reason}")
            }
            Self::UnknownState { sequence, position } => write!(
                formatter,
                "Markov sequence {sequence} state {position} is outside the finite vocabulary"
            ),
            Self::CountOverflow => write!(formatter, "Markov transition count overflow"),
            Self::DuplicateStateLabel => {
                write!(formatter, "Markov stable state labels must be unique")
            }
        }
    }
}

impl Error for MarkovError {}

fn score_sequences<S: Ord + Clone>(
    model: &MarkovModel<S>,
    sequences: &[Vec<S>],
    partition: &'static str,
) -> Result<TransitionScore, MarkovError> {
    let mut transitions = 0_u64;
    let mut log_likelihood = 0.0;
    for (sequence_index, sequence) in sequences.iter().enumerate() {
        for (position, state) in sequence.iter().enumerate() {
            model.require_state(state, sequence_index, position)?;
        }
        for pair in sequence.windows(2) {
            log_likelihood += model.transition_probability(&pair[0], &pair[1])?.ln();
            increment(&mut transitions)?;
        }
    }
    if transitions == 0 {
        return Err(MarkovError::NoTransitions { partition });
    }
    let mean_negative_log_likelihood = -log_likelihood / transitions as f64;
    Ok(TransitionScore {
        transitions,
        log_likelihood,
        mean_negative_log_likelihood,
        perplexity: mean_negative_log_likelihood.exp(),
    })
}

fn require_transitions<S>(
    sequences: &[Vec<S>],
    partition: &'static str,
) -> Result<(), MarkovError> {
    if sequences.iter().all(|sequence| sequence.len() < 2) {
        return Err(MarkovError::NoTransitions { partition });
    }
    Ok(())
}

fn increment(value: &mut u64) -> Result<(), MarkovError> {
    *value = value.checked_add(1).ok_or(MarkovError::CountOverflow)?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}
