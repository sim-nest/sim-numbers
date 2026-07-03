# sim-lib-numbers-float

In one line: It provides compact decimal numbers that trade a little accuracy for smaller size and speed.

## What it gives you

When you have a great many decimal values and do not need the fullest precision, a lighter decimal kind pays off. This gives you single-precision decimals: half the storage of the standard kind, which means more of them fit in memory and they move faster. You get the ordinary arithmetic on them, and whenever more accuracy is called for, they rise into the fuller decimal kind without any manual step. It is the right choice for large collections of measurements or graphics-style data where being compact matters more than the last few digits.

## Why you will be glad

- Large piles of decimal data take up less space and process quicker.
- You keep the option to widen into full precision the instant you need it.
- Memory-heavy work becomes lighter without changing how you write the math.

## Where it fits

This is the compact decimal domain of the SIM number stack. It sits just below the standard decimal kind and promotes smoothly into it, giving the constellation a lightweight option for bulk numeric data while staying fully compatible with the rest of the number family.
