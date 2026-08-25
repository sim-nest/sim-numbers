# Trace limiting boundaries without losing the fold

`sim-lib-numbers-continuation` follows a residual manifold through turning
points using secant/tangent prediction and a bordered Newton corrector. Every
accepted point carries its residual, tangent, normal, corrector report, and fold
classification; rejected predictions and adaptive step decisions remain
reviewable. Work, domain, and loop limits are explicit, so traces are bounded
and a recorded pair of points can resume deterministically.
