# sim-lib-numbers-arith

In one line: It lets you add, subtract, multiply, and divide numbers of different kinds together and always get the right answer.

## What it gives you

Real work mixes number types: a whole number here, a fraction there, a decimal somewhere else. This handles the everyday operations across all of them so you never have to stop and convert by hand. When you combine two different kinds, it quietly widens both to a shared kind that can hold the result without losing anything, then does the sum. If one of your inputs is a symbol rather than a concrete value, it hands the work to the algebra layer instead of failing. Plus, minus, times, divide, and running totals over a list all just work.

## Why you will be glad

- Mixing whole numbers, fractions, and decimals never surprises you with a wrong result.
- You skip the tedious, bug-prone job of converting types before every operation.
- Symbolic inputs are welcomed and routed onward instead of rejected.

## Where it fits

This is the common arithmetic counter of the SIM number stack, the place ordinary math requests land. It sits above the individual number domains and knows how to bring any two of them together, making it the shared entry point that keeps calculation consistent across the whole constellation.
