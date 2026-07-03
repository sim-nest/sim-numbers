# sim-lib-numbers-codec

In one line: It is how a new kind of number announces itself so the system knows how to read and write it.

## What it gives you

When someone wants to add a fresh kind of number to the system, that new kind needs a way to introduce itself: this is my name, here is how to recognize me when I appear in text, here is how to show me back. This provides the tidy paperwork for that introduction. It builds the small description a number provider hands to the runtime so the runtime can accept the new kind everywhere, reading it from what people type and printing it back out. The registration is uniform, so every number kind joins on equal footing.

## Why you will be glad

- Adding a new number kind is a clean, described step rather than a scramble.
- Every kind of number is recognized and displayed by the same consistent path.
- The system stays open to growth without special-casing each newcomer.

## Where it fits

This is the front-desk registration for number kinds in the SIM constellation. It connects the number domains to the wider codec surface, where text is turned into values and values back into text, keeping the whole family readable and writable through one shared arrangement.
