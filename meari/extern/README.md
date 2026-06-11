# External Codes Used

Originally written by Hojin Cho on 2025-01-07.

### PocketFFT (`extern.fft`)
(Written by Martin Reinecke) Downloaded on 2025-01-07 from the [repository](https://gitlab.mpcdf.mpg.de/mtr/pocketfft). Source codes redistributed under 3-caluse BSD. Original LICENSE.md is also presented in `fft/src/LICENSED.md`.

A very rough estimation suggests that `cfft` is faster than `numpy.fft.fft` by factor of 1.5x. 
- `n=1024`
- `cfft: 4.94 μs ± 27.5 ns` including overheads
- `numpy.fft.fft: 7.5 μs ± 37.2 ns per loop` including overheads

Note that there is a bug in `rfft` related functions in real input where the imaginary part of the first argument disappears. Reason is unclear, but advised not to use the `rfft` and stick to `cfft`.

### RandomNumberGenerator (`extern.rng`)
Wrapper for various RNG routines in C++. Custom wrapper class for RNGs in C++ is provided in `rng_wrapper.hpp`, which is written by H. Cho. Primary RNG engine is `PCG32`, which is included in PCG-C++ library, while other engines include Mersenne Twister 19937 (`MT19937`) and its 64bit variant (`MT19937_64`) defined in C++11 standard library `<random>`. Considering implementing `PCG64-DXSM` which is also defined in PCG-C++ headers and is to be implemented to `numpy`'s `default_rng` in future releases (currently `PCG64` is used, which has [a serious statistical issue under parallel processing](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html)).

PCG-C++ (written by M.E. O'Neill) is downloaded on 2025-01-14 from the [repository](https://github.com/imneme/pcg-cpp) (see also [the official website](https://www.pcg-random.org/)). Source codes are distributed under either of Apache 2.0 or MIT. A part of source codes are included in this project, and cython wrapper for this library is provided in `rng.pxd` by H. Cho, along with other interfaces. 

Benchmarks of generating 100,000,000 random numbers show that

- `PCG32` (800ms) and `numpy`'s `default_rng` has nearly identical performance (`PCG32` is ~2-3% faster.)
- `MT19937` is slower by ~40% than `PCG32`.
- `MTT9937_64` is slower by ~61% than `PCG32`.


### BLAS

See documentations in or [1](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/blas-routines.html), [2](https://www.seehuhn.de/pages/linear.html), or [SO](https://stackoverflow.com/q/3903879/4755229)

<!-- ### TinySpline (`extern.tinyspline`)

User should install via `conda install -c conda-forge libtinyspline`.

Refer to [the documentation](https://msteinbeck.github.io/tinyspline/), [C-interface API](https://msteinbeck.github.io/tinyspline/tinyspline_8h.html), [official github repo](https://github.com/msteinbeck/tinyspline), and [conda-forge feedstock](https://github.com/conda-forge/libtinyspline-feedstock).

Make sure to have `-ltinyspline` in `CC` command. -->