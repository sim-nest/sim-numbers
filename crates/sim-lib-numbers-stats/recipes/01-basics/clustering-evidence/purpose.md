# Seeded clustering with model-selection evidence

This generated two-cluster specimen demonstrates deterministic k-means++ with
four bounded restarts and regularized diagonal Gaussian-mixture EM. Both results
retain convergence, work, seed, repair, and termination evidence; the mixture
also retains AIC and BIC so callers can compare component counts honestly.

The runtime equivalents are:

```lisp
(stats/kmeans points :k 2 :control {:work 20000 :results 4 :seed 4})
(stats/gmm points :components 2 :covariance 'diagonal :regularization 1e-6)
```

Clustering owns unordered point partitions only. A caller that needs temporal
alignment or a staged state path composes `dynamic_time_warp` or
`layered_shortest_path` from `sim-lib-discrete-graph`; statistics does not carry
a private sequence-DP implementation.
