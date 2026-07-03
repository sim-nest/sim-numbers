# sim-lib-numbers-rk

In one line: It follows how a changing system moves over time, step by careful step.

## What it gives you

Many things are described not by where they are but by how they change: how fast something cools, how a population grows, how an object swings. Given that description, this traces the actual path forward through time. It advances the system in small increments, and it offers both steady fixed-size steps and adaptive ones that shorten where the motion is delicate and lengthen where it is calm, so you get accuracy where it matters and speed where it does not. The result is a faithful trajectory you can follow from a starting point onward as the system evolves.

## Why you will be glad

- You turn a rule about change into a concrete path you can watch unfold.
- Adaptive stepping keeps accuracy high through the tricky parts automatically.
- You choose between simple fixed steps and smarter adaptive ones to fit the job.

## Where it fits

This provides the time-stepping solvers for the numerical surface of the SIM number stack. It registers as one of the interchangeable backends for advancing changing systems, so the constellation can follow evolving-in-time problems without committing to a single stepping method.
