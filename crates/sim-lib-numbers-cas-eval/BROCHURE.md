# sim-lib-numbers-cas-eval

In one line: It takes a symbolic formula and works out its value once you supply what the unknowns stand for.

## What it gives you

You hold a formula with names like x and y in it, and eventually you want an answer. This is the part that plugs in the values you provide and computes the result. It can run in two ways: give it every value and it produces a concrete number, or leave some blanks and it keeps those parts symbolic, returning a partly-worked expression instead of complaining. An unknown you never filled in survives as itself rather than causing an error. That flexibility means you can evaluate as much as you know now and finish the rest later.

## Why you will be glad

- You get concrete answers the moment you supply the values.
- Missing values do not break things; the unknown parts simply stay as symbols.
- The same formula serves both full calculation and partial, exploratory work.

## Where it fits

This is the evaluation engine of the SIM computer-algebra layer. It bridges symbolic expressions and their everyday numeric meaning, letting the constellation move a formula from its abstract form into real results whenever the inputs are ready.
