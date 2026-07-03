# sim-lib-numbers-quad

In one line: It measures the area under a curve and estimates slopes by clever, careful sampling.

## What it gives you

When you need the total accumulated by a quantity, the area beneath its curve, this works it out by sampling the curve and adding up the pieces. It offers both steady, evenly-spaced approaches and smarter adaptive ones that spend extra effort only where the curve is tricky, so you get accuracy without wasted work. It also estimates how fast a quantity is changing by comparing values at nearby points. These are the practical tools for integration and slope-finding when the shape is known only by the values it produces, and you can choose the rule that fits your precision and cost.

## Why you will be glad

- You get dependable area-under-the-curve totals from just the values themselves.
- Adaptive rules concentrate effort where it counts, saving time on the easy stretches.
- Slope estimates come from the same handy toolkit, ready when you need them.

## Where it fits

This supplies integration and slope-estimation methods to the numerical surface of the SIM number stack. It plugs in as one of the interchangeable solver backends, so the constellation's numeric layer can offer area and rate-of-change calculations without hard-wiring a single technique.
