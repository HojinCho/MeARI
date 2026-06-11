from libc.stdint cimport uint64_t, uint32_t
cdef extern from "iostream":
    cdef cppclass ostream
    cdef cppclass istream
cdef extern from "pcg/pcg_random.hpp" :
    cdef cppclass pcg32:
        pcg32() noexcept nogil
        pcg32(uint64_t seed) noexcept nogil
        pcg32(uint64_t seed, uint64_t seq) noexcept nogil
        void seed(uint64_t seed) noexcept nogil
        void seed(uint64_t seed, uint64_t seq) noexcept nogil
        void discard(uint64_t n) noexcept nogil
        uint32_t min() noexcept nogil
        uint32_t max() noexcept nogil
        uint32_t operator()() noexcept nogil
        uint32_t operator()(uint32_t upper_bound) noexcept nogil
        bint operator==(const pcg32& rhs) noexcept nogil
        bint operator!=(const pcg32& rhs) noexcept nogil
        ostream& operator<<(ostream& out, const pcg32& rng) noexcept nogil
        istream& operator>>(istream& out, pcg32& rng) noexcept nogil
        # void advance(uint64_t delta) noexcept nogil