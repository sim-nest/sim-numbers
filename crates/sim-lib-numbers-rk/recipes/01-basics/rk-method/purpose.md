# Runge-Kutta ODE solver (descriptor)

The `numbers/rk` DOP853 solver integrates an ODE by bounded eighth-order adaptive Runge-Kutta stepping (here
exponential growth). The stepping loop runs outside the sandbox eval stack, so this recipe
documents the solver surface rather than running the integration live. Its method
receipt records accepted and rejected steps, RHS work, and the achieved local error.
