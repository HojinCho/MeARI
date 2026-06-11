# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, exp, cos, sin, pow,
    acos, asin,
    fmin, fmax, fabs,
)

from ..extern.lia cimport (
    lia_v_norm_A_i, lia3d_v_rho_R,
)

cdef double twoPI = 2*M_PI

cdef void assign_azimuthal(long nazm, double *azm) noexcept nogil:
    cdef double dazm = twoPI/nazm
    cdef long i
    for i in prange(nazm, nogil=True):
        azm[i] = i*dazm

cdef void assign_radial_lin(long nrad, double rad_in, double rad_out, double *rad) noexcept nogil:
    cdef double dr = (rad_out - rad_in)/nrad
    cdef long i
    for i in prange(nrad+1, nogil=True):
        rad[i] = rad_in + i*dr

cdef void assign_radial_log(long nrad, double rad_in, double rad_out, double *rad) noexcept nogil:
    cdef double dr = log(rad_out/rad_in)/nrad
    cdef long i
    for i in prange(nrad+1, nogil=True):
        rad[i] = rad_in*exp(i*dr)

cdef void __compute_norm(long nsize, double * vecarr, double * outarr) noexcept nogil:
    lia_v_norm_A_i(outarr, vecarr, 3, nsize)

cdef void __compute_rho_and_R(long nsize, double * xyz, double * rhoarr, double * Rarr) noexcept nogil:
    lia3d_v_rho_R(rhoarr, Rarr, xyz, nsize)

cdef void compute_lamppost_distance(long nsize, double * outdist, double * xyz0, double * rho, double H_lamp) noexcept nogil:
    # Make sure to use xyz0 in AGN coordinates
    cdef long i, I
    cdef double zmH
    for i in prange(nsize, nogil=True):
        zmH = xyz0[3*i+2] - H_lamp
        outdist[i] = sqrt(rho[i]*rho[i] + zmH*zmH)