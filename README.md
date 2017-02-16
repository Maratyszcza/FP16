# FP16
Header-only library for conversion to/from half-precision floating point formats

## Features

- Supports IEEE and ARM alternative half-precision floating-point format
    - Property converts infinities and NaNs
    - Properly converts denormal numbers, even on systems without denormal support
- Header-only library, no installation or build required
- Compatible with C99 and C++11
- Covered with unit tests and microbenchmarks
