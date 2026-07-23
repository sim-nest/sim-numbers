# sim-lib-numbers-tensor

In one line: It gives you grids of numbers, from simple lists to multi-dimensional blocks, as one kind of value.

## What it gives you

Lots of real data comes in arrangements: a row of readings, a table of figures, an image made of pixels, a stack of layers. This gives you a single, uniform way to hold any such arrangement of numbers, from a plain list up to a many-dimensioned block, and to build them with simple constructors for vectors, matrices, and beyond. Because the shape is part of the value, everything that works on grids works the same regardless of how many dimensions you use. It is the common container that turns scattered numbers into structured data you can operate on as a whole, and `tensor/cast` makes dtype conversion explicit when storage precision changes.

## Why you will be glad

- One consistent container handles lists, tables, and higher-dimensional data alike.
- Building vectors, matrices, and larger blocks is quick and uniform.
- Working with grouped numbers as a single value beats juggling them one by one.

## Where it fits

This is the grid-of-numbers domain at the center of the SIM tensor stack. Specialized backends for particular element types plug into the shared interface it defines, so it is the common ground that lets the constellation store and manipulate structured numeric data of any shape.
