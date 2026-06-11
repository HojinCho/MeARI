# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython cimport view
from cython.parallel import prange
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy

from .numpy_interface cimport numpy_ascontiguousarray, numpy_ravel_C, numpy_dbl_1d, numpy_dbl_2d, numpy_dbl_3d
# from libc.stdio  cimport printf

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow, exp,
    fmin, fmax, fabs, ceil, log2,
)

from ..physics.thermal cimport Planck_Responsivity_um
from .interpolation cimport ppi3_eval_bulk

#####################################################
# Testing Routines

cdef void __ppi3_stress_test(
    long niter, long nx, double * x, double * y,
    long n_bulk, double * x_rb, double * coef, long n_rb, 
) noexcept nogil:
    cdef long i, j, k
    cdef double * y_i
    for k in range(niter):
        for i in range(nx):
            y_i = y+i*n_bulk
            ppi3_eval_bulk(y_i, n_bulk, x[i], x_rb, coef, n_rb)
            for j in range(n_bulk):
                y_i[j] = exp(y_i[j])
    return

# cdef void __ppi3_fast_stress_test(
#     long niter, long nx, double * x, double * y,
#     long n_bulk, double * x_rb, double * coef, long n_rb, 
# ) noexcept nogil:
#     cdef long i, j, k
#     cdef double * y_i
#     for k in range(niter):
#         for i in range(nx):
#             y_i = y+i*n_bulk
#             ppi3_fast_eval_bulk(y_i, n_bulk, x[i], x_rb, coef, n_rb)
#             for j in range(n_bulk):
#                 y_i[j] = exp(y_i[j])
#     return

cdef void __monochrome_stress_test(
    long niter, long nx, double * x, double * y,
    long n_bulk, double * wavelength,
) noexcept nogil:
    cdef long i, j, k
    cdef double * y_i
    # Planck_Responsivity_um(double wave, double T)
    for k in range(niter):
        for i in range(nx):
            y_i = y+i*n_bulk
            for j in range(n_bulk):
                y_i[j] = Planck_Responsivity_um(wavelength[j], exp(x[i]))
    return

# numpy_ascontiguousarray, numpy_ravel_C, numpy_dbl_1d, numpy_dbl_2d
cpdef ppi3_stress_test(object x_np, object x_rb_np, object coef_np, long niter):
    if niter<1:
        raise ValueError("niter must be at least 1")
    xcont = numpy_ascontiguousarray(x_np)
    x_rb_cont = numpy_ascontiguousarray(x_rb_np)
    coef_cont = numpy_ascontiguousarray(coef_np)
    cdef double [:] x_c = xcont              # (nx)
    cdef double [:] x_rb_c = x_rb_cont       # (n_rb)
    cdef double [:, :, :] coef_mv = coef_cont # (n_rb+1, ndim, 4)
    cdef double [:] coef_c = numpy_ravel_C(coef_mv)
    cdef long nx = x_c.size
    cdef long n_rb = x_rb_c.size
    cdef long ndim = coef_mv.shape[1]
    cdef double * y = <double *> malloc(nx*ndim*sizeof(double))
    with nogil:
        __ppi3_stress_test(niter, nx, &x_c[0], y, ndim, &x_rb_c[0], &coef_c[0], n_rb)
    return numpy_dbl_2d(y, nx, ndim, True)

# cpdef ppi3_fast_stress_test(object x_np, object x_rb_np, object coef_np, long niter):
#     if niter<1:
#         raise ValueError("niter must be at least 1")
#     xcont = numpy_ascontiguousarray(x_np)
#     x_rb_cont = numpy_ascontiguousarray(x_rb_np)
#     coef_cont = numpy_ascontiguousarray(coef_np)
#     cdef double [:] x_c = xcont              # (nx)
#     cdef double [:] x_rb_c = x_rb_cont       # (n_rb)
#     cdef double [:, :, :] coef_mv = coef_cont # (n_rb+1, ndim, 4)
#     cdef double [:] coef_c = numpy_ravel_C(coef_mv)
#     cdef long nx = x_c.size
#     cdef long n_rb = x_rb_c.size
#     cdef long ndim = coef_mv.shape[1]
#     cdef double * y = <double *> malloc(nx*ndim*sizeof(double))
#     with nogil:
#         __ppi3_fast_stress_test(niter, nx, &x_c[0], y, ndim, &x_rb_c[0], &coef_c[0], n_rb)
#     return numpy_dbl_2d(y, nx, ndim, True)

cpdef monochrome_stress_test(object x_np, object w_np, long niter):
    if niter<1:
        raise ValueError("niter must be at least 1")
    xcont = numpy_ascontiguousarray(x_np)
    wcont = numpy_ascontiguousarray(w_np)
    cdef double [:] x_c = xcont # nx
    cdef double [:] w_c = wcont # ndim
    cdef long nx = x_c.size
    cdef long ndim = w_c.size
    cdef double * y = <double *> malloc(nx*ndim*sizeof(double))
    with nogil:
        __monochrome_stress_test(niter, nx, &x_c[0], y, ndim, &w_c[0])
    return numpy_dbl_2d(y, nx, ndim, True)
