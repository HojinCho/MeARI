# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

# Intended to convert malloc'd pointers into numpy array.
# Important: These functions causes potential memory leaks. 
#   Make sure if this the returned numpy array needs to be freed or not.
#   If it is intended to be a peek into an existing member of a class, 
#   e.g., coordinates of Mesh, use copy=False.
#   Otherwise, if it is newly malloc'd solely for returning to Python,
#   use copy=True.
# For Cython and NumPy native ways, see:
#   https://stackoverflow.com/a/23873586/4755229
#   https://stackoverflow.com/a/55959886/4755229

import numpy as __numpy__
cimport numpy as __numpy__

from libc.string cimport memcpy
from libc.stdlib cimport free

cpdef object numpy_ascontiguousarray(object arr):
    return __numpy__.ascontiguousarray(arr)

cpdef object numpy_ravel_C(object arr):
    return __numpy__.ravel(arr, order='C')

# No-copying routines.
# For peeking into properties as numpy array.
cdef object numpy_dbl_1d_nocopy(double * x, size_t n1):
    return __numpy__.array(<double [:n1]>x, copy=False, dtype=__numpy__.float64)
cdef object numpy_dbl_2d_nocopy(double * x, size_t n1, size_t n2):
    return __numpy__.array(<double [:n1,:n2]>x, copy=False, dtype=__numpy__.float64)
cdef object numpy_dbl_3d_nocopy(double * x, size_t n1, size_t n2, size_t n3):
    return __numpy__.array(<double [:n1,:n2,:n3]>x, copy=False, dtype=__numpy__.float64)
cdef object numpy_lng_1d_nocopy(long * x, size_t n1):
    return __numpy__.array(<long [:n1]>x, copy=False, dtype=__numpy__.int64)
cdef object numpy_lng_2d_nocopy(long * x, size_t n1, size_t n2):
    return __numpy__.array(<long [:n1,:n2]>x, copy=False, dtype=__numpy__.int64)
cdef object numpy_uch_1d_nocopy(unsigned char * x, size_t n1):
    return __numpy__.array(<unsigned char [:n1]>x, copy=False, dtype=__numpy__.uint8)
cdef object numpy_cmpl_1d_nocopy(double complex * x, size_t n1):
    return __numpy__.array(<double complex [:n1]>x, copy=False, dtype=__numpy__.complex128)

# Copying routines, with freeing the memory.
# For the pointers that are created specifically for returning to Python.


cdef object numpy_dbl_1d_copy_and_free(double * x, size_t n1):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.float64)
    cdef __numpy__.ndarray out = __numpy__.empty(n1, 
                order='C', dtype=__numpy__.float64)
    memcpy(<void *> &out.data[0], <void *>x, n1*sizeof(double))
    free(x)
    return out
cdef object numpy_dbl_2d_copy_and_free(double * x, size_t n1, size_t n2):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.float64)
    cdef __numpy__.ndarray out = __numpy__.empty((n1, n2), 
                order='C', dtype=__numpy__.float64)
    memcpy(<void *> &out.data[0], <void *>x, n1*n2*sizeof(double))
    free(x)
    return out
cdef object numpy_dbl_3d_copy_and_free(double * x, size_t n1, size_t n2, size_t n3):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.float64)
    cdef __numpy__.ndarray out = __numpy__.empty((n1, n2, n3), 
                order='C', dtype=__numpy__.float64)
    memcpy(<void *> &out.data[0], <void *>x, n1*n2*n3*sizeof(double))
    free(x)
    return out
cdef object numpy_lng_1d_copy_and_free(long * x, size_t n1):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.int64)
    cdef __numpy__.ndarray out = __numpy__.empty(n1, 
                order='C', dtype=__numpy__.int64)
    memcpy(<void *> &out.data[0], <void *>x, n1*sizeof(long))
    free(x)
    return out
cdef object numpy_lng_2d_copy_and_free(long * x, size_t n1, size_t n2):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.int64)
    cdef __numpy__.ndarray out = __numpy__.empty((n1, n2), 
                order='C', dtype=__numpy__.int64)
    memcpy(<void *> &out.data[0], <void *>x, n1*n2*sizeof(long))
    free(x)
    return out
cdef object numpy_uch_1d_copy_and_free(unsigned char * x, size_t n1):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.uint8)
    cdef __numpy__.ndarray out = __numpy__.empty(n1, 
                order='C', dtype=__numpy__.uint8)
    memcpy(<void *> &out.data[0], <void *>x, n1*sizeof(unsigned char))
    free(x)
    return out
cdef object numpy_cmpl_1d_copy_and_free(double complex * x, size_t n1):
    if x==NULL:
        return __numpy__.empty(0, dtype=__numpy__.complex128)
    cdef __numpy__.ndarray out = __numpy__.empty(n1, 
                order='C', dtype=__numpy__.complex128)
    memcpy(<void *> &out.data[0], <void *>x, n1*sizeof(double complex))
    free(x)
    return out


# Frontend

cdef object numpy_dbl_1d(double * x, size_t n1, bint copy):
    if copy:
        return numpy_dbl_1d_copy_and_free(x, n1)
    else:
        return numpy_dbl_1d_nocopy(x, n1)

cdef object numpy_dbl_2d(double * x, size_t n1, size_t n2, bint copy):
    if copy:
        return numpy_dbl_2d_copy_and_free(x, n1, n2)
    else:
        return numpy_dbl_2d_nocopy(x, n1, n2)

cdef object numpy_dbl_3d(double * x, size_t n1, size_t n2, size_t n3, bint copy):
    if copy:
        return numpy_dbl_3d_copy_and_free(x, n1, n2, n3)
    else:
        return numpy_dbl_3d_nocopy(x, n1, n2, n3)


cdef object numpy_lng_1d(long * x, size_t n1, bint copy):
    if copy:
        return numpy_lng_1d_copy_and_free(x, n1)
    else:
        return numpy_lng_1d_nocopy(x, n1)

cdef object numpy_lng_2d(long * x, size_t n1, size_t n2, bint copy):
    if copy:
        return numpy_lng_2d_copy_and_free(x, n1, n2)
    else:
        return numpy_lng_2d_nocopy(x, n1, n2)

cdef object numpy_uch_1d(unsigned char * x, size_t n1, bint copy):
    if copy:
        return numpy_uch_1d_copy_and_free(x, n1)
    else:
        return numpy_uch_1d_nocopy(x, n1)

cdef object numpy_cmpl_1d(double complex * x, size_t n1, bint copy):
    if copy:
        return numpy_cmpl_1d_copy_and_free(x, n1)
    else:
        return numpy_cmpl_1d_nocopy(x, n1)


from libc.stdlib cimport malloc

def numpy_empty_dbl_1d(size_t n):
    cdef double * x = <double *> malloc (n*sizeof(double))
    return numpy_dbl_1d(x, n, True)