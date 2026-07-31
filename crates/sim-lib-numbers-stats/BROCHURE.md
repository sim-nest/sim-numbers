# sim-lib-numbers-stats

In one line: It summarizes your data with averages, spreads, probabilities, and fairness checks.

## What it gives you

Point it at a set of numbers and it tells you the story they hold: the typical value, how spread out they are, and the shape of their likelihood. It covers the common descriptive summaries and probability helpers you reach for when making sense of data. Its finite Markov estimator keeps exact transition counts, smoothing, held-out likelihood, deterministic serialization, and corpus provenance open for inspection. It also computes fairness measures that flag when outcomes fall unevenly across groups. Results carry their evidence, so you can see not just the number but why it came out that way.

## Why you will be glad

- You get clear summaries of your data without assembling the math yourself.
- Learned transition scores remain reproducible counts and policy, never opaque weights.
- Fairness measures make uneven outcomes visible rather than hidden.
- Every result shows its supporting evidence, so the answer is checkable, not opaque.

## Where it fits

This is the statistics and probability layer of the SIM number stack. Built for decimal data, it gives the constellation a way to summarize, reason about likelihood, and audit fairness, all while keeping the justification for each figure open to inspection.
