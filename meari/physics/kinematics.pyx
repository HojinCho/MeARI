# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strlen, strcmp, strcpy, memcpy, memset

from libc.stdio cimport printf

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow, exp,
    fmin, fmax, fabs,
)

# from ..utils.types cimport Numeric, Integer, Real

from ..utils.numpy_interface cimport (
    numpy_dbl_1d,
    numpy_dbl_2d,
    numpy_lng_1d,
    numpy_lng_2d,
    numpy_uch_1d,
)

from ..mesh.transformables cimport rotate_vector, rotate_vector_T, inner_prod_const

################################################################################
# Physical Constants
################################################################################

## Exact-value Fundamental Constants
################################################################################
cdef double c0 = 29979245800.0       # cm/s  # Vacuum Speed of Light

## Measured Constants
################################################################################
cdef double GMSun = 1.32712440041279419e+26 # cm^3/s^2 double precision # https://ssd.jpl.nasa.gov/astro_par.html

################################################################################
# Unit Conversion Constants
################################################################################
cdef double cm_to_ltday = 3.86069554627490798E-16 # lt-day/cm

################################################################################
# Derived Constants
################################################################################
cdef double K = 2.263542193560001269 # 1e-5*sqrt(GMSun*cm_to_ltday)
# Highest Precision Number for K is 2.2635421935600012689134115779022

cdef void vel_field_disk_thick( 
    # Goad et al. 2012 MNRAS 426, 3086-3111, eq.2
    # Reduces to Keplerian if coords[3*i+2]==0 for all i.
    long n, 
    double logM, # Mass of the central object in solar masses, in common log.
    double * coords, double * R, # in lt-days
    double * vel, # output pointer in km/s
) noexcept nogil:
    cdef long i, I
    cdef double X = K*exp(logM*M_LN10)
    cdef double V
    for i in prange(n, nogil=True):
        I = 3*i
        V = X/sqrt(R[i]*R[i]*R[i])
        vel[I  ] = -coords[I+1]*V
        vel[I+1] =  coords[I  ]*V
        vel[I+2] = 0.