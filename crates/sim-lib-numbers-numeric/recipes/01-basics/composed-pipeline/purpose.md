# Composed pipeline

This recipe records the first-class numeric pipeline surface: define a function,
compose it with `rk4` for ODE solving, run the result, inspect the result table,
then compose a function with `simpson` for quadrature and compare the numeric
result.
