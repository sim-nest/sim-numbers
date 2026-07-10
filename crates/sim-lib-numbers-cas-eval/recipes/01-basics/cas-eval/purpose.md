# Symbolic CAS evaluation (descriptor)

Evaluating a CAS expression against a symbolic environment resolves bindings and folds
the expression tree through the computer-algebra engine -- machinery outside the sandbox
eval stack. The runnable `numbers/cas` recipe demonstrates live CAS construction; this
documents the `eval-cas` / symbolic-environment surface.
