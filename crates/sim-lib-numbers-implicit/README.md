# sim-lib-numbers-implicit

Three-stage, fifth-order Radau IIA integration for stiff ODEs and declared
index-1 mass-matrix or residual DAEs. The solver retains Newton, Jacobian,
linear-solve, rejection, dense-output, event, and work evidence.

The mathematical form is explicit: ODE problems provide `y' = f(t,y)`, mass
problems provide `M(t,y)y' = f(t,y)`, and residual problems provide
`F(t,y,y') = 0`. A residual problem must declare which variables are
differential and algebraic and must pass the index-1 stage Jacobian rank check.
