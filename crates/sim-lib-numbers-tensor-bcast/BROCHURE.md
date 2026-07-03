# sim-lib-numbers-tensor-bcast

In one line: It lets you combine grids of different shapes sensibly, stretching the smaller to match the larger.

## What it gives you

Often you want to apply one small thing across a big one: add a single number to every cell, or combine a short row with each row of a table. This handles those combinations by the familiar broadcasting rules, matching up shapes and stretching where it makes sense so the operation just works. It applies math across grids cell by cell, whether both sides are full grids or one is a lone value spread over the whole. The upshot is that you write the natural operation and it figures out how the shapes should line up, sparing you manual reshaping and looping.

## Why you will be glad

- Combining differently-shaped grids follows intuitive, widely-known rules.
- A single number or short row spreads across a whole grid without hand-written loops.
- You express the operation you mean and let the shape-matching sort itself out.

## Where it fits

This adds shape-aware, cell-by-cell operations to the SIM tensor stack. It layers onto the base grid domain without introducing new commands of its own, giving the constellation the everyday convenience of combining grids whose shapes do not exactly match.
