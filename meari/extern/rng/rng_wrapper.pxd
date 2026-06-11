from libc.stdint cimport uint_fast32_t, uint_fast64_t

cdef extern from "rng_wrapper.hpp":
    cdef cppclass RandomNumberGeneratorWrapper[T]:
        RandomNumberGeneratorWrapper(uint_fast32_t seed) noexcept nogil
        RandomNumberGeneratorWrapper(uint_fast64_t seed, uint_fast64_t seq) noexcept nogil
        double gen_normal(double mu, double sig) noexcept nogil
        void gen_normal_arr(double mu, double sig, double* vec, size_t size) noexcept nogil
