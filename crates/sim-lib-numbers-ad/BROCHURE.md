# sim-lib-numbers-ad

In one line: It works out how fast a calculation changes without you ever doing calculus by hand.

## What it gives you

When you have a formula and you want to know how sensitive its answer is to each input, this gives you those slopes automatically and exactly. It tracks the rate of change alongside every value as your computation runs, so you get precise derivatives instead of the shaky estimates you would get from nudging numbers and comparing. It handles both quick single-input cases and large calculations with many inputs at once, and it works across the different kinds of numbers the system understands. You write the formula once; the sensitivity information comes along for free.

## Why you will be glad

- You get exact slopes, not noisy approximations that drift with your choice of step size.
- Sensitivity analysis and optimization stop being error-prone hand work.
- The same approach scales from one input to thousands without changing how you write things.

## Where it fits

This is the sensitivity engine underneath the number stack in the SIM constellation. Other parts that fit curves, train models, or search for best answers lean on it to learn which direction to move. It quietly supplies the derivative information that turns a plain calculation into one that can be tuned and improved.
