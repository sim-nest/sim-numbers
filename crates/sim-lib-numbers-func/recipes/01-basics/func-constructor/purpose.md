# Apply a function value

Construct the function `x -> x + 1` as `#(numbers/Func (x) (+ x 1))` and apply it to `5`.
The func domain evaluates the call and returns `6` -- a first-class function value, built
and applied through the runtime.
