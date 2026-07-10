# Symbolic differentiation (descriptor)

Symbolic differentiation rewrites a CAS expression into its derivative. It dispatches
through the computer-algebra engine, which the sandbox eval stack does not drive. The
runnable `numbers/cas` cas-constructor recipe shows live CAS construction; this recipe
documents the `diff` / symbolic-derivative surface.
