# Rerun a kernel at extended precision

Load `numbers/extended`, retain the numerical kernel's existing `RealScalar`
implementation, and select `DoubleDouble` at its scalar boundary. Compare the
result with the binary64 run and an exact rational oracle. Preserve the
canonical `dd:<hi-bits>:<lo-bits>` literal when recording evidence so the low
component is never rounded through decimal text.
