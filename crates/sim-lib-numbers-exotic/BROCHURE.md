# sim-lib-numbers-exotic

In one line: It represents numbers to unlimited precision by unfolding them only as far as you actually need.

## What it gives you

Some numbers, like certain famous constants, have digits that never end. This holds such values in a form that can, in principle, go on forever, yet it only computes the detail you ask for. You can request an ordinary decimal approximation and it will unfold just enough to give you one, without ever pretending the value was finite in the first place. Built-in constants come ready to use. It is a way to keep endlessly-precise real numbers around and draw from them on demand, so accuracy is a dial you turn rather than a limit you hit.

## Why you will be glad

- Never-ending values are kept honestly, not silently rounded from the start.
- You draw exactly as much precision as your task needs, and no more.
- Handy built-in constants are available without you constructing them.

## Where it fits

This is the home for unusual number kinds in the SIM number stack, currently the endlessly-unfolding real numbers. It broadens the constellation beyond fixed-size values, offering a precise alternative for work where ordinary decimals would quietly lose accuracy.
