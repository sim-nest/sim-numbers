# sim-lib-numbers-fixed

In one line: It offers whole numbers in specific sizes, so you can match the exact range and storage a task needs.

## What it gives you

Not every count needs the same amount of room. Sometimes a small, tightly-bounded number is exactly right; other times you want a wider one. This gives you a whole family of sized whole numbers, both the ones that allow negatives and the ones that only count upward, across a range of widths. You pick the size that fits, and values step up into larger sizes along a clear widening path when a calculation grows. This lets you be deliberate about how much range a number holds, which matters when you care about compact storage or matching an outside format.

## Why you will be glad

- You choose exactly how much range and storage each whole number uses.
- Both signed and unsigned counts are covered by one consistent family.
- Values widen along a predictable path as calculations demand more room.

## Where it fits

This supplies the sized whole-number domains of the SIM number stack. Alongside the general-purpose integer kinds, it gives the constellation precise control over number width, which is valuable when interoperating with external data formats or keeping memory tight.
