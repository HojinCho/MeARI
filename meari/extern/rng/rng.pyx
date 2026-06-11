# distutils: language = c++
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

from libc.stdint cimport uint_fast32_t, uint_fast64_t
from libc.stdlib cimport malloc, free

from .rng cimport RngType
from .rng.rng_wrapper cimport RandomNumberGeneratorWrapper
from .rng.pcg cimport pcg32
from libcpp.random cimport mt19937, mt19937_64


from ..utils.numpy_interface cimport numpy_dbl_1d

cdef double __gen_normal_mt19937(void* rng, double mu, double sig) noexcept nogil:
    return (<RandomNumberGeneratorWrapper[mt19937]*>rng).gen_normal(mu, sig)
cdef void __gen_normal_arr_mt19937(void* rng, double mu, double sig, double* vec, size_t size) noexcept nogil:
    (<RandomNumberGeneratorWrapper[mt19937]*>rng).gen_normal_arr(mu, sig, vec, size)
cdef void destructor_mt19937(void* rng) noexcept nogil:
    cdef RandomNumberGeneratorWrapper[mt19937] * temp = <RandomNumberGeneratorWrapper[mt19937] *> rng
    del temp

cdef double __gen_normal_mt19937_64(void* rng, double mu, double sig) noexcept nogil:
    return (<RandomNumberGeneratorWrapper[mt19937_64]*>rng).gen_normal(mu, sig)
cdef void __gen_normal_arr_mt19937_64(void* rng, double mu, double sig, double* vec, size_t size) noexcept nogil:
    (<RandomNumberGeneratorWrapper[mt19937_64]*>rng).gen_normal_arr(mu, sig, vec, size)
cdef void destructor_mt19937_64(void* rng) noexcept nogil:
    cdef RandomNumberGeneratorWrapper[mt19937_64] * temp = <RandomNumberGeneratorWrapper[mt19937_64] *> rng
    del temp

cdef double __gen_normal_pcg32(void* rng, double mu, double sig) noexcept nogil:
    return (<RandomNumberGeneratorWrapper[pcg32]*>rng).gen_normal(mu, sig)
cdef void __gen_normal_arr_pcg32(void* rng, double mu, double sig, double* vec, size_t size) noexcept nogil:
    (<RandomNumberGeneratorWrapper[pcg32]*>rng).gen_normal_arr(mu, sig, vec, size)
cdef void destructor_pcg32(void* rng) noexcept nogil:
    cdef RandomNumberGeneratorWrapper[pcg32] * temp = <RandomNumberGeneratorWrapper[pcg32] *> rng
    del temp

cdef class RandomNumberGenerator:
    def __cinit__(self, RngType rng_type=RngType.RNG_PCG32, uint_fast64_t seed=0, uint_fast64_t seq=0):
        with nogil:
            if rng_type == RngType.RNG_PCG32:
                self._rng_type = rng_type
                self._rng = <void *> (new RandomNumberGeneratorWrapper[pcg32](seed, seq))
                self.__gen_normal_func = &__gen_normal_pcg32
                self.__gen_normal_arr_func = &__gen_normal_arr_pcg32
                self.__destructor = &destructor_pcg32
            elif rng_type == RngType.RNG_MT19937:
                self._rng_type = rng_type
                self._rng = <void *> (new RandomNumberGeneratorWrapper[mt19937](seed))
                self.__gen_normal_func = &__gen_normal_mt19937
                self.__gen_normal_arr_func = &__gen_normal_arr_mt19937
                self.__destructor = &destructor_mt19937
            elif rng_type == RngType.RNG_MT19937_64:
                self._rng_type = rng_type
                self._rng = <void *> (new RandomNumberGeneratorWrapper[mt19937_64](seed))
                self.__gen_normal_func = &__gen_normal_mt19937_64
                self.__gen_normal_arr_func = &__gen_normal_arr_mt19937_64
                self.__destructor = &destructor_mt19937_64
            else:
                raise ValueError("Unknown RNG type")
    def __dealloc__(self):
        with nogil:
            self.__destructor(self._rng)
    
    # Add resampling method? (for CCCD or other bootstrapping purpose)

    cdef double _gen_normal(self, double mu, double sig) noexcept nogil:
        return self.__gen_normal_func(self._rng, mu, sig)
    cdef void _gen_normal_arr(self, double mu, double sig, double* vec, size_t size) noexcept nogil:
        self.__gen_normal_arr_func(self._rng, mu, sig, vec, size)

    def normal(self, shape=None, double mu=0, double sig=1):
        if shape is None:
            return self._gen_normal(mu, sig)
        if not hasattr(shape, '__len__'):
            shape = (shape,)
        cdef size_t i, ndim = len(shape)
        cdef size_t size = 1
        for i in range(ndim):
            size *= int(shape[i])
        cdef double * vec = <double *> malloc(size*sizeof(double))
        self._gen_normal_arr(mu, sig, vec, size)
        return numpy_dbl_1d(vec, size, True).reshape(shape)
            
def RNG(rng=None, seed=0, seq=0): # Only for debug purposes
    rngdict = {
        'pcg32'      : RngType.RNG_PCG32,
        'mt19937'    : RngType.RNG_MT19937,
        'mt19937_64' : RngType.RNG_MT19937_64,
    }
    default_rng = 'pcg32'
    if rng is None:
        rng = default_rng
    display_keys = [x for x in rngdict.keys()]
    for i, key in enumerate(display_keys):
        if key==default_rng:
            display_keys[i] = "'"+key+"' (default)"
        else:
            display_keys[i] = "'"+key+"'"
    if not isinstance(rng, str):
        raise ValueError(
            "Must provide a string to specify RNG, not a %s" % type(rng)+"\n"+\
            "Allowed values are: "+", ".join(display_keys)
        )
    rng = rng.lower()
    if rng not in rngdict:
        raise ValueError(
            "Unknown RNG type: %s" % rng+"\n"+\
            "Allowed values are: "+", ".join(display_keys)
        )
    return RandomNumberGenerator(rngdict[rng], seed, seq)

    