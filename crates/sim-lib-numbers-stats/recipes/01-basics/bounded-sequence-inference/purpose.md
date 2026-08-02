# Bounded Sequence Inference (descriptor)

The generic statistics owner provides mergeable streaming quantiles with an
exact small-data reference and a hard retained-entry policy. It also provides
finite discrete- and Gaussian-emission hidden Markov models, normalized
forward/backward inference, Viterbi and posterior decoding, and deterministic
Baum-Welch fitting.

The runnable example merges two quantile summaries, checks normalized HMM
inference, and fits a categorical model. The fitting report records its
initialization seed, likelihood history,
convergence, iteration and work counts, numerical repairs, and termination
reason. Domain adapters supply their own state and symbol meanings, so the
example uses deliberately generic labels rather than a domain vocabulary.
