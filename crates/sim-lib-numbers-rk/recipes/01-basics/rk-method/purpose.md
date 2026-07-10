# Runge-Kutta ODE solver (descriptor)

The `numbers/rk` RKF45 solver integrates an ODE by adaptive Runge-Kutta stepping (here
exponential growth). The stepping loop runs outside the sandbox eval stack, so this recipe
documents the solver surface rather than running the integration live.
