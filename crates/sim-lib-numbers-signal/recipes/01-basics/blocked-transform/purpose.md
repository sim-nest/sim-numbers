# Bounded multidimensional transform

Apply a separable two-axis FFT once in canonical memory and once through a
caller-supplied Table block store. The plan admits execution only when one
transform line plus one block fits the declared scratch ceiling, then records
passes, scratch bytes, block I/O, precision, and a content digest. No
filesystem path is part of the storage contract.
