# sim-lib-numbers-stats

In one line: It summarizes streams and infers finite sequences without hiding memory, error, or convergence.

## What it gives you

Point it at a set of numbers and it tells you the story they hold: the typical value, how spread out they are, and the shape of their likelihood. Seeded k-means and regularized Gaussian mixtures expose their centroids, assignments, covariance, convergence, repair, restart, work, AIC, and BIC evidence. Mergeable streaming quantiles state their rank-error and memory policy and stay exact for small inputs. Finite Markov and hidden Markov models expose their transition and emission rows, while normalized forward/backward, Viterbi, posterior decoding, and bounded Baum-Welch retain likelihood, convergence, repairs, work, seed, and termination evidence. It also computes fairness measures that flag when outcomes fall unevenly across groups.

## Why you will be glad

- You get clear summaries of your data without assembling the math yourself.
- Streaming summaries refuse memory overruns instead of quietly weakening their error policy.
- Learned sequence models remain inspectable and reproducible, never opaque weights.
- Cluster searches and mixture fits are reproducible under seed and fail closed on non-finite or singular data.
- Fairness measures make uneven outcomes visible rather than hidden.
- Every result shows its supporting evidence, so the answer is checkable, not opaque.

## Where it fits

This is the statistics and probability layer of the SIM number stack. It supplies generic summaries, point clustering, and finite-state inference; domain adapters keep their own state and symbol meanings outside the numbers repository, while temporal alignment remains composed from the discrete graph owner.
