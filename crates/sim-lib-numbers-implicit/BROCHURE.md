# Stiff integration without hidden promises

`sim-lib-numbers-implicit` advances stiff ODEs and genuine index-1 DAEs with
three-stage, fifth-order Radau IIA. Every result explains where its Jacobian
came from, why it was rebuilt, how often its factor was reused, whether Newton
failed, and what rank and conditioning the accepted linear systems exhibited.

Collocation polynomials provide continuous output and event location. Dense
output is never manufactured from an unconverged stage, while singular
Jacobians and work exhaustion are explicit refusals rather than dubious values.
