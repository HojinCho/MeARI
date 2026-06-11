# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, realloc, calloc, free
from libc.string cimport strlen, strcmp, strcpy, memcpy

from ..utils.numpy_interface cimport (
    numpy_dbl_1d,
    numpy_dbl_2d,
)

###
# General-purpose n-dimensional physical vector space based on Cython.
# In theory, this can be used even for pseudo-metric spaces, such as Minkowski metric.
# May add dyads and higher rank tensors if necessary.
# May need to implement BLAS and LAPACK for matrix operations?
###

cdef void rotate_vector(long ndata, long ndim, double *rotmat, double * vec_in, double * vec_out) noexcept nogil
cdef void rotate_vector_T(long ndata, long ndim, double *rotmat, double * vec_in, double * vec_out) noexcept nogil
cdef void inner_prod_const(long ndata, long ndim, double * vec, double * vec_const, double * out) noexcept nogil
cdef void inner_prod(long ndata, long ndim, double * vec1, double * vec2, double * out) noexcept nogil
cdef void inner_prod_xyz(long ndata, long ndim, double * vec, double * xyz, double * R, double * out) noexcept nogil
cdef void generalized_inner_prod(long ndata, long ndim, double * metric, double * vec1, double * vec2, double * out) noexcept nogil

cdef class StrLists():
    cdef:
        public long N
        long *len
        char **data
        long __current_buffer_len
    cpdef void add(self, str s_str) noexcept
    cpdef void remove(self, str s_str) noexcept
    cpdef bint is_in(self, str s_str) noexcept
    cpdef long get_index(self, str s_str) noexcept
    cpdef str get(self, long i)
    cpdef list to_list(self)

@cython.final
cdef class Scalars():
    cdef:
        public long N
        public long ndata
        size_t size
        long __current_buffer_len
        StrLists entries
        double ** __data
    cpdef object get_by_index(self, long i)
    cpdef object get(self, str s_str)
    cpdef void allocate_entry(self, str s_str) noexcept nogil
    cpdef void idxset(self, long idx, double [:] data) noexcept nogil
    cdef void idxset_ptr(self, long idx, double * data) noexcept nogil
    cpdef void set(self, str s_str, double [:] data) noexcept nogil
    cdef void set_ptr(self, str s_str, double * data) noexcept nogil
    cpdef void unset(self, str s_str) noexcept nogil

@cython.final
cdef class Vectors():
    cdef:
        public long N
        public long ndata
        public long ndim
        long nd2
        size_t size
        long __current_buffer_len
        StrLists entries
        double ** __data_agn_coord
        double ** __data_obs_coord
        double * rotmat
    cpdef void rotate(self, double [:] rotmat) noexcept nogil
    cdef void rotate_ptr(self, double * rotmat) noexcept nogil
    cpdef object get_agn(self, str s_str)
    cpdef object get_obs(self, str s_str)
    cpdef object get_agn_by_index(self, long i)
    cpdef object get_obs_by_index(self, long i)
    cpdef void allocate_entry(self, str s_str) noexcept nogil
    cpdef void idxset_in_agn_coord(self, long idx, double [:,:] data) noexcept nogil
    cpdef void idxset_in_obs_coord(self, long idx, double [:,:] data) noexcept nogil
    cdef void idxset_in_agn_ptr(self, long idx, double * data) noexcept nogil
    cdef void idxset_in_obs_ptr(self, long idx, double * data) noexcept nogil
    cpdef void set_in_agn_coord(self, str s_str, double [:,:] data) noexcept nogil
    cpdef void set_in_obs_coord(self, str s_str, double [:,:] data) noexcept nogil
    cdef void set_in_agn_ptr(self, str s_str, double * data) noexcept nogil
    cdef void set_in_obs_ptr(self, str s_str, double * data) noexcept nogil
    cpdef void unset(self, str s_str) noexcept nogil