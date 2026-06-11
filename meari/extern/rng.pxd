

from libc.stdint cimport uint_fast32_t, uint_fast64_t, uint64_t, uint32_t
cdef extern from "iostream":
    cdef cppclass ostream
    cdef cppclass istream

cdef enum RngType:
    RNG_MT19937
    RNG_MT19937_64
    RNG_PCG32

cdef class RandomNumberGenerator:
    cdef:
        void * _rng
        RngType _rng_type
        double (*__gen_normal_func)(void*, double, double) noexcept nogil
        void (*__gen_normal_arr_func)(void*, double, double, double*, size_t) noexcept nogil
        void (*__destructor)(void *) noexcept nogil

    cdef double _gen_normal(self, double mu, double sig) noexcept nogil
    cdef void _gen_normal_arr(self, double mu, double sig, double* vec, size_t size) noexcept nogil
